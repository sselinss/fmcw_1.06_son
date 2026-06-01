"""
Canli FMCW goruntuleyici — Pi4.

3 panel:
  1. Skor-zaman trendi  : hareket skoru (dB), esik cizgileriyle — "biri var mi?"
  2. Range profili      : baseline ustu hareket (dB) vs menzil + CFAR peak
  3. Waterfall          : zaman x menzil isi haritasi

UDP 5006 dinler. Calistirma (Pi'de):
    DISPLAY=:0 python3 plotter.py
"""

import argparse
import json
import socket
import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

PORT      = 5006
MAX_R     = 10.0
HISTORY   = 100         # waterfall frame
TREND_LEN = 200         # skor trendinde gosterilecek nokta
MIN_DB    = 8.0         # ALIVE (medium) esigi
STRONG_DB = 15.0        # ALIVE (high)  esigi

_latest = {}
_lock   = threading.Lock()


def _listener(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    print(f"[plotter] UDP dinleniyor :{port} ...")
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            with _lock:
                _latest.update(json.loads(data))
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[plotter] paket hatasi: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    threading.Thread(target=_listener, args=(args.port,), daemon=True).start()

    fig = plt.figure(figsize=(11, 8), facecolor="#0d0d0d")
    fig.canvas.manager.set_window_title("FMCW Radar — Canli")
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.1, 1.3, 1.0],
                           left=0.10, right=0.96, top=0.93, bottom=0.07, hspace=0.5)

    ax_trend = fig.add_subplot(gs[0])
    ax_prof  = fig.add_subplot(gs[1])
    ax_wfall = fig.add_subplot(gs[2])

    def style(ax):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5)

    # ── Panel 1: skor trendi ──────────────────────────────────────────────
    style(ax_trend)
    ax_trend.set_ylim(0, 30)
    ax_trend.set_xlim(0, TREND_LEN)
    ax_trend.set_ylabel("Skor (dB)", color="#aaaaaa")
    ax_trend.set_xlabel("Zaman (kare)", color="#aaaaaa")
    line_trend, = ax_trend.plot([], [], color="#00ff88", linewidth=1.5)
    ax_trend.axhline(MIN_DB,    color="#ffaa00", linestyle="--", linewidth=0.8, label="Medium")
    ax_trend.axhline(STRONG_DB, color="#ff4444", linestyle="--", linewidth=0.8, label="High")
    ax_trend.legend(facecolor="#111111", labelcolor="#cccccc", fontsize=8, loc="upper right")
    title_text = ax_trend.set_title("Bekleniyor...", color="#ffffff", fontsize=12)

    # ── Panel 2: range profili ────────────────────────────────────────────
    style(ax_prof)
    ax_prof.set_xlim(0, MAX_R)
    ax_prof.set_ylim(-10, 30)
    ax_prof.set_xlabel("Menzil (m)", color="#aaaaaa")
    ax_prof.set_ylabel("Hareket (dB)", color="#aaaaaa")
    line_prof,  = ax_prof.plot([], [], color="#00aaff", linewidth=1.2, label="Range profili")
    scat_peaks  = ax_prof.scatter([], [], color="#ff4444", zorder=5, s=70, label="Tespit")
    ax_prof.legend(facecolor="#111111", labelcolor="#cccccc", fontsize=8)

    # ── Panel 3: waterfall ────────────────────────────────────────────────
    style(ax_wfall)
    ax_wfall.set_xlabel("Menzil (m)", color="#aaaaaa")
    ax_wfall.set_ylabel("Frame", color="#aaaaaa")

    # --- veri tamponlari ---
    trend_buf  = deque(maxlen=TREND_LEN)
    wfall_data = deque(maxlen=HISTORY)
    wfall_img  = [None]
    fc = [0]

    def update(_):
        with _lock:
            pkt = dict(_latest)
        if not pkt or "profile_db" not in pkt:
            return

        ranges  = np.array(pkt["range_m"])
        profile = np.array(pkt["profile_db"])
        peaks   = pkt.get("peaks", [])
        dec     = pkt.get("decision", "?")
        br      = pkt.get("breathing_bpm", 0.0)
        hr      = pkt.get("heartbeat_bpm", 0.0)
        idx     = pkt.get("cpi_index", "?")
        score   = pkt.get("score_db", 0.0)

        # Panel 1: skor trendi
        trend_buf.append(score)
        ty = list(trend_buf)
        line_trend.set_data(np.arange(len(ty)), ty)
        col = "#00ff88" if dec == "ALIVE" else "#888888"
        line_trend.set_color(col)

        alive_str = "** CANLI TESPIT **" if dec == "ALIVE" else "tespit yok"
        tcol = "#00ff88" if dec == "ALIVE" else "#ff5555"
        title_text.set_text(
            f"[{idx}]  {alive_str}   skor={score:.1f} dB   "
            f"Nefes: {br:.0f} bpm   Kalp: {hr:.0f} bpm"
        )
        title_text.set_color(tcol)

        # Panel 2: range profili
        mask = ranges <= MAX_R
        r_cut, p_cut = ranges[mask], profile[mask]
        line_prof.set_data(r_cut, p_cut)
        if peaks:
            px = [p["range_m"]  for p in peaks if p["range_m"] <= MAX_R]
            py = [p.get("power_db", 0.0) for p in peaks if p["range_m"] <= MAX_R]
            scat_peaks.set_offsets(np.c_[px, py] if px else np.empty((0, 2)))
        else:
            scat_peaks.set_offsets(np.empty((0, 2)))

        # Panel 3: waterfall
        row = np.interp(np.linspace(0, MAX_R, 256),
                        r_cut if len(r_cut) else [0, MAX_R],
                        p_cut if len(p_cut) else np.zeros(2))
        wfall_data.append(row)
        fc[0] += 1
        if fc[0] % 3 == 0:
            mat = np.array(wfall_data)
            if wfall_img[0] is None:
                wfall_img[0] = ax_wfall.imshow(
                    mat, aspect="auto", origin="lower",
                    extent=[0, MAX_R, 0, HISTORY], vmin=0, vmax=20, cmap="inferno")
            else:
                wfall_img[0].set_data(mat)
                wfall_img[0].set_extent([0, MAX_R, 0, len(wfall_data)])

    ani = FuncAnimation(fig, update, interval=120, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
