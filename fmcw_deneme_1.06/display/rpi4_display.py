"""
RPi4 görüntüleme: Pluto'dan gelen JSON sonuçlarını dinle, ekranda göster.

Çalıştırma (RPi4'te):
    python3 rpi4_display.py
    python3 rpi4_display.py --port 5005 --proto udp
"""

from __future__ import annotations
import argparse
import json
import socket
import sys
import time


# Pygame varsa grafik, yoksa terminal modu
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


# ---------------------------------------------------------------------------
# Terminal görüntüleyici
# ---------------------------------------------------------------------------

class TerminalDisplay:
    def update(self, data: dict):
        tgts = data.get("targets", [])
        rng  = f"{tgts[0]['range_m']:.2f} m" if tgts else "—"
        dec  = data.get("decision", "?")
        br   = data.get("breathing_bpm", 0.0)
        hr   = data.get("heartbeat_bpm", 0.0)
        idx  = data.get("cpi_index", "?")

        color = "\033[92m" if dec == "ALIVE" else "\033[91m"
        rst   = "\033[0m"

        print(
            f"\r[CPI {idx:>5}] {color}{dec:<5}{rst}  "
            f"Menzil: {rng:>8}  "
            f"Nefes: {br:5.1f} bpm  "
            f"Kalp: {hr:5.1f} bpm      ",
            end="", flush=True,
        )


# ---------------------------------------------------------------------------
# Pygame görüntüleyici (RPi4'te ekran varsa)
# ---------------------------------------------------------------------------

class PygameDisplay:
    W, H = 800, 480

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("FMCW Radar — Canlı")
        self._font_big = pygame.font.SysFont("monospace", 72, bold=True)
        self._font_med = pygame.font.SysFont("monospace", 36)
        self._font_sml = pygame.font.SysFont("monospace", 24)
        self._clock    = pygame.time.Clock()

    def update(self, data: dict):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        tgts = data.get("targets", [])
        rng  = f"{tgts[0]['range_m']:.2f} m" if tgts else "—"
        dec  = data.get("decision", "?")
        br   = data.get("breathing_bpm", 0.0)
        hr   = data.get("heartbeat_bpm", 0.0)

        alive  = dec == "ALIVE"
        bg     = (10, 30, 10)  if alive else (30, 10, 10)
        fg     = (0, 255, 80)  if alive else (255, 60, 60)

        self._screen.fill(bg)

        # Büyük karar yazısı
        surf = self._font_big.render(dec, True, fg)
        self._screen.blit(surf, (self.W // 2 - surf.get_width() // 2, 60))

        # Menzil
        surf = self._font_med.render(f"Menzil : {rng}", True, (220, 220, 220))
        self._screen.blit(surf, (60, 200))

        # Vital signs
        surf = self._font_med.render(f"Nefes  : {br:.1f} bpm", True, (180, 220, 255))
        self._screen.blit(surf, (60, 270))

        surf = self._font_med.render(f"Kalp   : {hr:.1f} bpm", True, (255, 180, 180))
        self._screen.blit(surf, (60, 340))

        # Hedef listesi
        for i, t in enumerate(tgts[:3]):
            line = f"  Hedef {i+1}: {t['range_m']:.2f} m  {t['doppler_hz']:+.1f} Hz"
            surf = self._font_sml.render(line, True, (160, 160, 160))
            self._screen.blit(surf, (60, 420 + i * 24))

        pygame.display.flip()
        self._clock.tick(30)


# ---------------------------------------------------------------------------
# UDP / TCP alıcı
# ---------------------------------------------------------------------------

def listen(port: int, proto: str):
    if proto == "tcp":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        print(f"[display] TCP dinleniyor :{port} …")
        conn, addr = srv.accept()
        print(f"[display] Bağlantı: {addr}")
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield json.loads(line)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        print(f"[display] UDP dinleniyor :{port} …")
        while True:
            data, _ = sock.recvfrom(65535)
            yield json.loads(data)


def main():
    ap = argparse.ArgumentParser(description="FMCW Radar RPi4 Görüntüleyici")
    ap.add_argument("--port",  type=int, default=5005)
    ap.add_argument("--proto", default="udp", choices=["udp", "tcp"])
    ap.add_argument("--no-gui", action="store_true", help="Pygame'i devre dışı bırak")
    args = ap.parse_args()

    if _HAS_PYGAME and not args.no_gui:
        disp = PygameDisplay()
        print("[display] Pygame modu")
    else:
        disp = TerminalDisplay()
        print("[display] Terminal modu")

    for packet in listen(args.port, args.proto):
        disp.update(packet)


if __name__ == "__main__":
    main()
