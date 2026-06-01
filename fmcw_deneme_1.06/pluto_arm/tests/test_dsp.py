"""
DSP pipeline birim testleri. Hardware gerekmez — synthetic veri kullanır.
Çalıştırma: python3 test_dsp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from radar_config import load_config
from acquisition import _synthetic_cpi
from dsp import range_doppler_map, detect_targets, process_cpi


def make_config():
    cfg = load_config()
    cfg["chirp"]["n_chirps"] = 64
    cfg["dsp"]["doppler_fft_size"] = 64
    # range_fft_size=2048 default kullan (spc=2000 ile uyumlu)
    return cfg


def test_range_doppler_shape():
    cfg = make_config()
    cpi = _synthetic_cpi(cfg)
    rd, ranges, dopplers = range_doppler_map(cpi, cfg)
    nr = cfg["dsp"]["range_fft_size"] // 2
    nd = cfg["dsp"]["doppler_fft_size"]
    assert rd.shape == (nd, nr), f"Expected {(nd, nr)}, got {rd.shape}"
    assert len(ranges)   == nr
    assert len(dopplers) == nd
    print(f"[PASS] range_doppler_map shape: {rd.shape}")


def test_range_axis_physical():
    cfg = make_config()
    cpi = _synthetic_cpi(cfg)
    _, ranges, _ = range_doppler_map(cpi, cfg)
    assert ranges[0] == 0.0
    assert ranges[-1] > 0.0
    print(f"[PASS] Range axis: [{ranges[0]:.2f}, {ranges[-1]:.2f}] m")


def test_target_detected():
    cfg = make_config()
    # Synthetic CPI ~3 m'de hedef içeriyor
    cpi = _synthetic_cpi(cfg)
    rd, ranges, dopplers = range_doppler_map(cpi, cfg)
    targets = detect_targets(rd, ranges, dopplers, cfg)
    assert len(targets) > 0, "No target detected"
    best = targets[0]
    assert 0.3 <= best["range_m"] <= 10.0
    print(f"[PASS] Target detected: {best['range_m']:.2f} m")


def test_process_cpi_keys():
    cfg = make_config()
    cpi = _synthetic_cpi(cfg)
    result = process_cpi(cpi, cfg)
    for key in ("targets", "breathing_bpm", "heartbeat_bpm", "decision"):
        assert key in result, f"Missing key: {key}"
    assert result["decision"] in ("ALIVE", "NONE")
    print(f"[PASS] process_cpi output: decision={result['decision']}")


if __name__ == "__main__":
    test_range_doppler_shape()
    test_range_axis_physical()
    test_target_detected()
    test_process_cpi_keys()
    print("\nAll tests passed.")
