"""
İşlenmiş radar sonuçlarını RPi4'e TCP/UDP üzerinden gönder.
Ham I/Q yerine sadece küçük JSON paketleri iletilir (~200 byte/frame).

Kullanım:
    sender = ResultSender(config)
    sender.send(result_dict)
    sender.close()
"""

from __future__ import annotations
import json
import socket
import time


class ResultSender:
    def __init__(self, config: dict, protocol: str = "udp"):
        disp = config["display"]
        self._ip       = disp["rpi4_ip"]
        self._port     = int(disp["rpi4_port"])
        self._interval = float(disp.get("send_interval_s", 0.5))
        self._proto    = protocol.lower()
        self._last_send = 0.0

        if self._proto == "tcp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self._ip, self._port))
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"[sender] {self._proto.upper()} → {self._ip}:{self._port}")

    def send(self, result: dict) -> bool:
        """
        Sonucu gönder. send_interval_s aralığına uymayan çağrılar atlanır.
        Başarı durumunda True döner.
        """
        now = time.monotonic()
        if now - self._last_send < self._interval:
            return False

        payload = json.dumps(result, separators=(",", ":")).encode() + b"\n"

        try:
            if self._proto == "tcp":
                self._sock.sendall(payload)
            else:
                self._sock.sendto(payload, (self._ip, self._port))
            self._last_send = now
            return True
        except OSError as e:
            print(f"[sender] gönderme hatası: {e}")
            return False

    def close(self):
        self._sock.close()
