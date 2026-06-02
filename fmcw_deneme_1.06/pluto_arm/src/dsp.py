"""
DSP pipeline: range FFT → Range-Doppler map → CFAR → vital signs.

Pluto'nun ARM'ında çalışacak şekilde optimize edilmiştir:
- NumPy tabanlı, scipy isteğe bağlı
- Hafif CFAR (1-D, range ekseni üzerinde)
- Nefes/kalp hızı basit FFT pik tespiti ile
"""

from __future__ import annotations
import numpy as np

_C = 3e8


# ---------------------------------------------------------------------------
# Pencere
# ---------------------------------------------------------------------------

def _window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name in ("rect", "none"):
        return np.ones(n, dtype=np.float32)
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.hanning(n).astype(np.float32)


# ---------------------------------------------------------------------------
# Range-Doppler haritası
# ---------------------------------------------------------------------------

def range_doppler_map(
    cpi: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CPI (n_chirps, spc) → Range-Doppler haritası + eksen etiketleri.

    Returns
    -------
    rd_map       : (n_doppler, n_range//2) complex64
    ranges_m     : (n_range//2,) float32
    dopplers_hz  : (n_doppler,) float32
    """
    n_chirps, spc = cpi.shape
    fs   = float(config["sdr"]["sample_rate"])
    B    = float(config["chirp"]["bandwidth"])
    T    = float(config["chirp"]["duration"])
    nr   = int(config["dsp"]["range_fft_size"])
    nd   = int(config["dsp"]["doppler_fft_size"])
    win  = config["dsp"].get("window", "hann")

    if nr < spc:
        # Chirp'i kırp: sadece ilk nr örneği kullan (yakın menzil, ARM belleği korur)
        spc = nr
        cpi = cpi[:, :spc]

    # Range FFT
    rw  = _window(win, spc)
    rfft = np.fft.fft(cpi * rw[np.newaxis, :], n=nr, axis=1)[:, :nr // 2]

    # Doppler FFT
    dw  = _window(win, n_chirps)
    rd  = np.fft.fftshift(
        np.fft.fft(rfft * dw[:, np.newaxis], n=nd, axis=0), axes=0
    ).astype(np.complex64)

    # Fiziksel eksenler
    ranges_m    = (np.arange(nr // 2) * (fs / nr) * (_C * T) / (2.0 * B)).astype(np.float32)
    dopplers_hz = np.fft.fftshift(np.fft.fftfreq(nd, d=T)).astype(np.float32)

    return rd, ranges_m, dopplers_hz


# ---------------------------------------------------------------------------
# 1-D CFAR (range ekseni, her Doppler bin için)
# ---------------------------------------------------------------------------

def cfar_1d(
    rd_mag: np.ndarray,
    guard: int = 2,
    train: int = 8,
    pfa: float = 1e-3,
) -> np.ndarray:
    """
    Vektörize CA-CFAR — her Doppler satırı için cumsum tabanlı.

    Returns
    -------
    mask : bool array, shape (n_doppler, n_range//2)
    """
    nd, nr = rd_mag.shape
    total  = 2 * train
    alpha  = total * (pfa ** (-1.0 / total) - 1.0)

    pad    = guard + train
    # Sıfır padding: kenar hücrelerinde training alanı küçülür, yanlış yansıma olmaz
    padded = np.pad(rd_mag.astype(np.float32), ((0, 0), (pad, pad)), mode="constant", constant_values=0)
    # cumsum along range axis; shape (nd, nr + 2*pad + 1)
    cs = np.concatenate([np.zeros((nd, 1), dtype=np.float32), np.cumsum(padded, axis=1)], axis=1)

    idx = np.arange(nr) + pad           # CUT indices in padded array
    left_sum  = cs[:, idx - guard]      - cs[:, idx - guard - train]
    right_sum = cs[:, idx + guard + train + 1] - cs[:, idx + guard + 1]
    noise_est = (left_sum + right_sum) / total

    return rd_mag > (alpha * noise_est)


def detect_targets(
    rd: np.ndarray,
    ranges_m: np.ndarray,
    dopplers_hz: np.ndarray,
    config: dict,
) -> list[dict]:
    """
    Range-Doppler haritasından hedefleri tespit et.

    Returns
    -------
    Liste: [{"range_m": float, "doppler_hz": float, "power": float}, ...]
    min/max_range_m filtresi uygulanmış.
    """
    cfar_cfg  = config["dsp"]["cfar"]
    min_r     = float(config["dsp"].get("min_range_m", 0.3))
    max_r     = float(config["dsp"].get("max_range_m", 10.0))

    mag  = np.abs(rd)
    mask = cfar_1d(
        mag,
        guard=cfar_cfg["guard_cells"],
        train=cfar_cfg["training_cells"],
        pfa=cfar_cfg["false_alarm_rate"],
    )

    hits = []
    d_idx, r_idx = np.where(mask)
    for d, r in zip(d_idx, r_idx):
        rng = float(ranges_m[r])
        if not (min_r <= rng <= max_r):
            continue
        hits.append({
            "range_m":    rng,
            "doppler_hz": float(dopplers_hz[d]),
            "power":      float(mag[d, r]),
        })

    # Güce göre sırala
    hits.sort(key=lambda x: x["power"], reverse=True)
    return hits


# ---------------------------------------------------------------------------
# Vital signs — phase tabanlı (slow-time)
# ---------------------------------------------------------------------------

def extract_slow_time(
    cpi: np.ndarray,
    config: dict,
    target_range_m: float,
) -> np.ndarray:
    """
    CPI'dan hedef menzilindeki slow-time fazını çıkar.

    Returns
    -------
    phase : (n_chirps,) float32  — unwrapped phase (radyan)
    """
    fs   = float(config["sdr"]["sample_rate"])
    B    = float(config["chirp"]["bandwidth"])
    T    = float(config["chirp"]["duration"])
    nr   = int(config["dsp"]["range_fft_size"])
    win  = config["dsp"].get("window", "hann")
    spc  = cpi.shape[1]

    if nr < spc:
        spc = nr
        cpi = cpi[:, :spc]

    rw   = _window(win, spc)
    rfft = np.fft.fft(cpi * rw[np.newaxis, :], n=nr, axis=1)

    # Hedef range bin
    k = int(round(target_range_m * 2 * B / (_C * T) * nr / fs))
    k = np.clip(k, 0, nr // 2 - 1)

    phase = np.unwrap(np.angle(rfft[:, k])).astype(np.float32)
    return phase


def _bandpass(sig: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Basit FIR bandpass (scipy varsa) veya FFT maskeleme."""
    try:
        from scipy.signal import firwin, lfilter
        n_taps = min(len(sig) // 4 * 2 + 1, 51)
        nyq = fs / 2.0
        b = firwin(n_taps, [low / nyq, high / nyq], pass_zero=False)
        return lfilter(b, 1.0, sig).astype(np.float32)
    except ImportError:
        # FFT maskeleme
        F = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
        F[(freqs < low) | (freqs > high)] = 0
        return np.fft.irfft(F, n=len(sig)).astype(np.float32)


def estimate_rate_hz(sig: np.ndarray, fs: float, band: list) -> float:
    """Sinyal içindeki dominant frekansı (Hz) döndür."""
    F = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    idx = np.argmax(F[mask])
    return float(freqs[mask][idx])


def vital_signs(
    phase: np.ndarray,
    config: dict,
    chirp_rate_hz: float,
) -> dict:
    vs = config["dsp"]["vital_signs"]
    br_band = vs["breathing_band_hz"]
    hr_band = vs["heartbeat_band_hz"]
    fs  = chirp_rate_hz
    nyq = fs / 2.0          # olcebilecegimiz en yuksek frekans

    # Nefes: band ust siniri Nyquist'in guvenli altindaysa olc, degilse 0
    if br_band[1] < 0.95 * nyq:
        br_sig = _bandpass(phase, fs, br_band[0], br_band[1])
        br_hz  = estimate_rate_hz(br_sig, fs, br_band)
    else:
        br_hz = 0.0

    # Kalp: kare hizi yeterli degilse olcme (cokme onlenir)
    if hr_band[1] < 0.95 * nyq:
        hr_sig = _bandpass(phase, fs, hr_band[0], hr_band[1])
        hr_hz  = estimate_rate_hz(hr_sig, fs, hr_band)
    else:
        hr_hz = 0.0

    return {
        "breathing_bpm":  round(br_hz * 60, 1),
        "heartbeat_bpm":  round(hr_hz * 60, 1),
    }


# ---------------------------------------------------------------------------
# Tek çağrıyla tam pipeline
# ---------------------------------------------------------------------------

def process_cpi(cpi: np.ndarray, config: dict) -> dict:
    """
    Bir CPI'ı alıp tam sonucu döndür.

    Returns
    -------
    dict: {
        "targets":        [{"range_m", "doppler_hz", "power"}, ...],
        "breathing_bpm":  float,
        "heartbeat_bpm":  float,
        "decision":       "ALIVE" | "NONE",
    }
    """
    rd, ranges_m, dopplers_hz = range_doppler_map(cpi, config)
    targets = detect_targets(rd, ranges_m, dopplers_hz, config)

    breathing_bpm = 0.0
    heartbeat_bpm = 0.0
    decision      = "NONE"

    if targets:
        best_range = targets[0]["range_m"]
        T          = float(config["chirp"]["duration"])
        chirp_rate = 1.0 / T

        phase = extract_slow_time(cpi, config, best_range)
        vs    = vital_signs(phase, config, chirp_rate)

        breathing_bpm = vs["breathing_bpm"]
        heartbeat_bpm = vs["heartbeat_bpm"]

        if breathing_bpm > 0 or heartbeat_bpm > 0:
            decision = "ALIVE"

    # Range profili: Doppler boyutunda max projeksiyon → dB normalize
    rd_mag   = np.abs(rd)
    profile  = np.max(rd_mag, axis=0).astype(np.float32)
    peak_val = float(np.max(profile)) if profile.max() > 0 else 1.0
    profile_db = 20 * np.log10(profile / peak_val + 1e-12)

    # Peak'lere dB degeri ekle
    for t in targets[:3]:
        bin_idx = int(round(t["range_m"] / float(ranges_m[1]))) if len(ranges_m) > 1 else 0
        bin_idx = np.clip(bin_idx, 0, len(ranges_m) - 1)
        t["power_db"] = float(profile_db[bin_idx])

    return {
        "targets":       targets[:3],
        "breathing_bpm": breathing_bpm,
        "heartbeat_bpm": heartbeat_bpm,
        "decision":      decision,
        "range_m":       ranges_m.tolist(),
        "profile_db":    profile_db.tolist(),
    }
