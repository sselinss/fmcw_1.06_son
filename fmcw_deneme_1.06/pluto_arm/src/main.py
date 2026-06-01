"""
FMCW Radar ana dongusu — RPi4 + Pluto USB.

Tespit mantigi (calisan referans live_operator.py'den):
  - Kareler arasi FAZ degisimi → oturan/nefes alan insani yakalar
    (tek CPI Doppler'i sifir verir; insan saniyeler icinde faz oynatir)
  - Magnitude baseline cikarma → statik clutter (duvar, TX kacagi) bastirilir
  - Vital signs: hedef bin'deki kompleks degerin zamanla biriktirilmesi

Calistirma:
    python3 main.py
    python3 main.py --stdout
"""

from __future__ import annotations
import argparse
import json
import socket
import time
from collections import deque

import numpy as np

from radar_config import load_config
from acquisition import make_radar
from dsp import vital_signs

# --- Tespit parametreleri (referanstan) ---
CALIBRATION_FRAMES = 15
MIN_MOTION_DB      = 8.0
STRONG_MOTION_DB   = 15.0
SIGNAL_FLOOR_FACTOR = 2.0
PHASE_MOTION_GAIN  = 20.0
IGNORE_NEAR_M      = 0.0     # bin 0 dahil — TX kacagi baseline ile cikarilir
RANGE_MAX_M        = 3.0     # Pluto + UWB icin gercekci insan menzili
VITAL_WINDOW_S     = 20.0


def phase_motion_score(ranges_m, cur, prev):
    """Kareler arasi kompleks degisimden hareket skoru (dB-benzeri)."""
    if prev is None:
        return 0.0, -1
    valid = (ranges_m >= IGNORE_NEAR_M) & (ranges_m <= RANGE_MAX_M)
    if not np.any(valid):
        return 0.0, -1
    mag = np.abs(cur)
    noise = float(np.median(mag)) + 1e-9
    signal = mag > (SIGNAL_FLOOR_FACTOR * noise)
    sel = valid & signal
    if not np.any(sel):
        return 0.0, -1
    ref = np.maximum(np.abs(prev), noise)
    rel = np.abs(cur - prev) / ref
    rel_masked = np.where(sel, rel, 0.0)
    best_bin = int(np.argmax(rel_masked))
    return float(rel_masked[best_bin]) * PHASE_MOTION_GAIN, best_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    radar  = make_radar(config)

    print("[main] Basliyor.")
    print(f"[main] Frekans   : {config['sdr']['center_freq']/1e9:.3f} GHz")
    print(f"[main] Bandwidth : {config['chirp']['bandwidth']/1e6:.1f} MHz")
    print(f"[main] Kalibrasyon: {CALIBRATION_FRAMES} kare (radar onunu bos tut)")
    print("[main] Ctrl+C ile durdur.")

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send_plot(payload):
        if args.stdout:
            return
        try:
            _sock.sendto(json.dumps(payload, separators=(",", ":")).encode(), ("127.0.0.1", 5006))
        except OSError:
            pass

    # Ilk kare
    first = None
    while first is None:
        first = radar.capture_frame()
    ranges_m, profile_db, complex_row = first

    baseline      = profile_db.copy()
    calib_count   = 1
    prev_complex  = None

    # Vital signs icin hedef bin'deki kompleks degeri biriktir
    vital_buf  = deque()        # (timestamp, complex_value)
    frame_count = 0
    breathing_bpm = 0.0
    heartbeat_bpm = 0.0
    t0 = time.monotonic()

    try:
        while True:
            frame = radar.capture_frame()
            if frame is None:
                continue
            ranges_m, profile_db, complex_row = frame
            frame_count += 1
            now = time.monotonic()

            # ── Kalibrasyon (ilk N kare baseline) ─────────────────────────
            if calib_count < CALIBRATION_FRAMES:
                baseline = (baseline * calib_count + profile_db) / (calib_count + 1)
                calib_count += 1
                prev_complex = complex_row
                print(f"[kalibrasyon {calib_count}/{CALIBRATION_FRAMES}] ...")
                continue

            # ── Magnitude hareket (baseline cikar) ────────────────────────
            motion_db = profile_db - baseline
            valid = (ranges_m >= IGNORE_NEAR_M) & (ranges_m <= RANGE_MAX_M)
            mag_score = float(np.max(motion_db[valid])) if np.any(valid) else 0.0

            # ── Faz hareketi (kareler arasi) ──────────────────────────────
            ph_score, best_bin = phase_motion_score(ranges_m, complex_row, prev_complex)
            prev_complex = complex_row

            score = max(mag_score, ph_score)

            # ── Vital signs birikimi (en guclu sinyal bin'i) ──────────────
            if best_bin >= 0:
                vital_buf.append((now, complex_row[best_bin]))
            # pencereyi kirp
            while vital_buf and (now - vital_buf[0][0]) > VITAL_WINDOW_S:
                vital_buf.popleft()

            # Vital signs hesapla (yeterli veri varsa, her ~1 s)
            if len(vital_buf) >= 32 and frame_count % 10 == 0:
                ts   = np.array([v[0] for v in vital_buf])
                vals = np.array([v[1] for v in vital_buf])
                fs_slow = len(ts) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else 10.0
                phase = np.unwrap(np.angle(vals)).astype(np.float32)
                vs = vital_signs(phase, config, fs_slow)
                breathing_bpm = vs["breathing_bpm"]
                heartbeat_bpm = vs["heartbeat_bpm"]

            # ── Karar ─────────────────────────────────────────────────────
            if score >= STRONG_MOTION_DB:
                decision, conf = "ALIVE", "high"
            elif score >= MIN_MOTION_DB:
                decision, conf = "ALIVE", "medium"
            else:
                decision, conf = "NONE", "low"

            target_range = float(ranges_m[best_bin]) if best_bin >= 0 else None
            rng_str = f"{target_range:.2f} m" if target_range else "   ---  "

            print(
                f"[{frame_count:5d}] {decision:<5s}({conf:<6s}) "
                f"skor={score:5.1f}dB  menzil={rng_str}  "
                f"nefes={breathing_bpm:5.1f}  kalp={heartbeat_bpm:5.1f}  "
                f"buf={len(vital_buf)}"
            )

            # ── Plotter ────────────────────────────────────────────────────
            # profil: baseline cikarilmis, gosterim icin normalize
            disp_prof = (profile_db - baseline)
            peaks = []
            if best_bin >= 0 and score >= MIN_MOTION_DB:
                peaks = [{"range_m": float(ranges_m[best_bin]),
                          "power_db": float(disp_prof[best_bin])}]

            send_plot({
                "cpi_index":     frame_count,
                "decision":      decision,
                "targets":       peaks,
                "peaks":         peaks,
                "breathing_bpm": breathing_bpm,
                "heartbeat_bpm": heartbeat_bpm,
                "range_m":       ranges_m.tolist(),
                "profile_db":    disp_prof.tolist(),
                "score_db":      round(score, 1),
            })

    except KeyboardInterrupt:
        print("\n[main] Durduruldu.")
    finally:
        radar.close()
        _sock.close()


if __name__ == "__main__":
    main()
