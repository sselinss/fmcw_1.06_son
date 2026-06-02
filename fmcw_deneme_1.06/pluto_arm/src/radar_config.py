"""
Radar config loader. Reads YAML, falls back to defaults if PyYAML not available.
"""

from __future__ import annotations
import os

_DEFAULT = {
    "sdr": {
        "pluto_uri":   "ip:192.168.2.1",
        "sample_rate": 20_000_000,
        "center_freq": 1_500_000_000,
        "rx_gain":     40,
        "tx_gain":     -10,
    },
    "chirp": {
        "bandwidth":  18_000_000,
        "duration":   1e-3,
        "n_chirps":   128,
    },
    "dsp": {
        "range_fft_size":   2048,
        "doppler_fft_size": 128,
        "window":           "hann",
        "min_range_m":      0.3,
        "max_range_m":      10.0,
        "cfar": {
            "guard_cells":      2,
            "training_cells":   8,
            "false_alarm_rate": 1e-3,
        },
        "vital_signs": {
            "breathing_band_hz": [0.1, 0.5],
            "heartbeat_band_hz": [0.8, 2.0],
        },
    },
    "display": {
        "rpi4_ip":        "192.168.1.19",
        "rpi4_port":      5005,
        "send_interval_s": 0.5,
    },
}


def load_config(path: str | None = None) -> dict:
    """
    Load radar config from a YAML file. Falls back to built-in defaults
    if the file is missing or PyYAML is unavailable (e.g. bare Pluto env).
    """
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        # Önce aynı dizine bak (Pi deploy), sonra proje köküne
        candidates = [
            os.path.join(here, "radar_params.yaml"),
            os.path.join(here, "..", "..", "configs", "radar_params.yaml"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    try:
        import yaml
        with open(path) as f:
            user = yaml.safe_load(f)
        # Deep-merge user values over defaults
        return _deep_merge(_DEFAULT, user)
    except FileNotFoundError:
        print(f"[config] {path} not found — using defaults")
        return dict(_DEFAULT)
    except ImportError:
        print("[config] PyYAML not installed — using defaults")
        return dict(_DEFAULT)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
