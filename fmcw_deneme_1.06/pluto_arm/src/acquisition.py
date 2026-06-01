"""
FMCW acquisition — ADALM-Pluto (pyadi-iio), kalici baglanti.

Calisan referans mimari (live_operator.py) birebir izlenir:
  1. Baglanti + config + cyclic TX → BIR KEZ (__init__)
  2. Her frame:
       - kucuk RX tamponu al (2 chirp boyu)
       - cross-correlation ile chirp baslangiclarini bul (hizalama!)
       - hizalanmis bloklari cikar
       - conj(chirp) ile dechirp → beat
       - range FFT (zero-pad) → range profili + complex_row
  3. complex_row: kareler arasi faz tutuldugu icin vital signs / hareket icin kullanilir

Pluto cyclic TX, RX yakalama ile senkron DEGIL — bu yuzden hizalama sart.
"""

from __future__ import annotations
import numpy as np

_C = 3e8
TX_SCALE = 2 ** 14
RANGE_FFT_PAD = 4   # range FFT zero-pad faktoru (bin granularitesi)


def _generate_chirp(config: dict) -> np.ndarray:
    fs = float(config["sdr"]["sample_rate"])
    B  = float(config["chirp"]["bandwidth"])
    T  = float(config["chirp"]["duration"])
    if B > fs:
        raise ValueError(f"bandwidth {B/1e6:.1f} MHz > sample_rate {fs/1e6:.1f} MHz")
    n  = int(round(fs * T))
    t  = np.arange(n) / fs
    k  = B / T
    phase = 2.0 * np.pi * (-B / 2.0 * t + 0.5 * k * t * t)
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------------------------------------------------------
# Hizalama (cross-correlation)
# ---------------------------------------------------------------------------

def _find_chirp_starts(rx, tx_chirp, threshold_ratio=0.3, min_sep=None):
    from scipy import signal as scisig
    if min_sep is None:
        min_sep = len(tx_chirp)
    corr = scisig.correlate(rx, tx_chirp, mode="valid")
    corr_mag = np.abs(corr)
    peak_max = float(np.max(corr_mag))
    if peak_max == 0:
        return np.empty(0, dtype=np.int64)
    peaks, _ = scisig.find_peaks(corr_mag, height=threshold_ratio * peak_max, distance=min_sep)
    return peaks.astype(np.int64)


def _extract_aligned_blocks(rx, tx_chirp, max_blocks=20):
    chirp_len = len(tx_chirp)
    starts = _find_chirp_starts(rx, tx_chirp)
    valid = starts[starts + chirp_len <= len(rx)]
    if max_blocks is not None:
        valid = valid[:max_blocks]
    if len(valid) == 0:
        return np.empty((0, chirp_len), dtype=rx.dtype)
    return np.stack([rx[s:s + chirp_len] for s in valid])


# ---------------------------------------------------------------------------
# Kalici Pluto FMCW arayuzu
# ---------------------------------------------------------------------------

class PlutoFMCW:
    """Baglanti + cyclic TX bir kez kurulur, frame'ler tek tek alinir."""

    def __init__(self, config: dict):
        import adi
        self.config = config
        self.fs = float(config["sdr"]["sample_rate"])
        self.B  = float(config["chirp"]["bandwidth"])
        self.T  = float(config["chirp"]["duration"])
        self.window = config["dsp"].get("window", "hann")
        self.chirp_len = int(round(self.fs * self.T))

        self.tx_chirp = _generate_chirp(config)

        uri     = config["sdr"]["pluto_uri"]
        fc      = int(config["sdr"]["center_freq"])
        rx_gain = float(config["sdr"]["rx_gain"])
        tx_gain = float(config["sdr"]["tx_gain"])
        rf_bw   = int(min(max(self.B * 1.2, 1_000_000), 56_000_000))

        sdr = adi.Pluto(uri)
        sdr.sample_rate       = int(self.fs)
        sdr.rx_rf_bandwidth   = rf_bw
        sdr.tx_rf_bandwidth   = rf_bw
        sdr.rx_lo             = fc
        sdr.tx_lo             = fc
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0   = rx_gain
        sdr.tx_hardwaregain_chan0   = tx_gain
        sdr.rx_buffer_size    = self.chirp_len * 2   # 2 chirp boyu

        # Cyclic TX — BIR KEZ
        sdr.tx_cyclic_buffer = True
        sdr.tx((self.tx_chirp * TX_SCALE).astype(np.complex64))

        self.sdr = sdr

        # Range ekseni (zero-pad'li)
        nfft = self.chirp_len * RANGE_FFT_PAD
        bin_freqs = np.arange(nfft // 2) * (self.fs / nfft)
        self.ranges_m = (bin_freqs * (_C * self.T) / (2.0 * self.B)).astype(np.float32)

    def _window(self, n):
        name = self.window.lower()
        if name in ("rect", "none"):  return np.ones(n, np.float32)
        if name == "hamming":         return np.hamming(n).astype(np.float32)
        return np.hanning(n).astype(np.float32)

    def capture_frame(self):
        """
        Bir frame yakala.

        Returns
        -------
        (ranges_m, profile_db, complex_row) veya None (hizalama basarisiz)
          ranges_m    : (nfft//2,) menzil ekseni (m)
          profile_db  : (nfft//2,) guc profili (dB)
          complex_row : (nfft//2,) koherent ortalama range FFT (faz korunur)
        """
        rx = self.sdr.rx()
        if rx.dtype != np.complex64:
            rx = rx.astype(np.complex64)

        blocks = _extract_aligned_blocks(rx, self.tx_chirp, max_blocks=20)
        if len(blocks) == 0:
            return None

        w = self._window(self.chirp_len)
        blocks = blocks * w[None, :]
        beat = blocks * np.conj(self.tx_chirp)[None, :]

        nfft = self.chirp_len * RANGE_FFT_PAD
        range_fft = np.fft.fft(beat, n=nfft, axis=1)[:, :nfft // 2]

        complex_row = np.mean(range_fft, axis=0).astype(np.complex64)
        profile_db  = (10.0 * np.log10(np.mean(np.abs(range_fft) ** 2, axis=0) + 1e-12)).astype(np.float32)

        return self.ranges_m, profile_db, complex_row

    def close(self):
        try:
            self.sdr.tx_destroy_buffer()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synthetic fallback (hardware yoksa)
# ---------------------------------------------------------------------------

class SyntheticFMCW:
    """Pluto yokken test icin: ~1.5 m'de nefes alan hedef simulasyonu."""

    def __init__(self, config: dict):
        self.config = config
        self.fs = float(config["sdr"]["sample_rate"])
        self.B  = float(config["chirp"]["bandwidth"])
        self.T  = float(config["chirp"]["duration"])
        self.fc = float(config["sdr"]["center_freq"])
        self.chirp_len = int(round(self.fs * self.T))
        self.frame = 0

        nfft = self.chirp_len * RANGE_FFT_PAD
        bin_freqs = np.arange(nfft // 2) * (self.fs / nfft)
        self.ranges_m = (bin_freqs * (_C * self.T) / (2.0 * self.B)).astype(np.float32)
        self.rng = np.random.default_rng()

    def capture_frame(self):
        nfft = self.chirp_len * RANGE_FFT_PAD
        n = nfft // 2
        self.frame += 1
        # Hedef ~1.5 m, nefes fazi
        target_range = 1.5
        k = int(round(target_range * 2 * self.B / (_C * self.T) * nfft / self.fs))
        k = np.clip(k, 0, n - 1)
        lam = _C / self.fc
        # nefes: 0.25 Hz, frame ~ 0.1 s aralikli varsay
        t = self.frame * 0.1
        disp = 4e-3 * np.sin(2 * np.pi * 0.25 * t)
        phase = 4 * np.pi * disp / lam
        complex_row = (0.05 * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))).astype(np.complex64)
        complex_row[k] += 2.0 * np.exp(1j * phase)
        profile_db = (20 * np.log10(np.abs(complex_row) + 1e-12)).astype(np.float32)
        return self.ranges_m, profile_db, complex_row

    def close(self):
        pass


def make_radar(config: dict):
    """Pluto varsa PlutoFMCW, yoksa SyntheticFMCW dondur."""
    try:
        import adi  # noqa
        return PlutoFMCW(config)
    except ImportError:
        print("[acquisition] pyadi-iio yok — synthetic mod")
        return SyntheticFMCW(config)
