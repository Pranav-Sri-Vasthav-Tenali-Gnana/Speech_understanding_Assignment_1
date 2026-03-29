import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import dct

def load_audio(path: str):
    """Return (signal, sample_rate) as float32 mono."""
    signal, sr = sf.read(path, always_2d=False)
    if signal.ndim == 2:          # stereo → mono
        signal = signal.mean(axis=1)
    return signal.astype(np.float32), sr

def pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(signal[0], signal[1:] - coef * signal[:-1])

def frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    indices = (
        np.tile(np.arange(frame_len), (n_frames, 1))
        + np.tile(np.arange(n_frames) * hop_len, (frame_len, 1)).T
    )
    return signal[indices]          # shape: (n_frames, frame_len)


def get_window(name: str, length: int) -> np.ndarray:
    name = name.lower()
    if name == "hamming":
        return np.hamming(length)
    elif name in ("hanning", "hann"):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))
    elif name in ("rectangular", "rect"):
        return np.ones(length)
    else:
        raise ValueError(f"Unknown window: {name!r}. Use hamming/hanning/rectangular.")

def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    return (1.0 / n_fft) * (mag ** 2)
def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    f_min: float = 0.0,
    f_max: float = None,
) -> np.ndarray:
    if f_max is None:
        f_max = sr / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left  = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            filters[m - 1, k] = (k - f_left) / (f_center - f_left + 1e-10)
        for k in range(f_center, f_right):
            filters[m - 1, k] = (f_right - k) / (f_right - f_center + 1e-10)

    return filters

def log_mel_energies(power_spec: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    energies = power_spec @ filterbank.T
    return np.log(energies + 1e-10)

def apply_dct(log_energies: np.ndarray, n_mfcc: int) -> np.ndarray:
    return dct(log_energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]

def delta_coefficients(features: np.ndarray, N: int = 2) -> np.ndarray:
    T = features.shape[0]
    denom = 2.0 * sum(n ** 2 for n in range(1, N + 1))
    delta = np.zeros_like(features)
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
    for t in range(T):
        delta[t] = sum(n * (padded[t + N + n] - padded[t + N - n]) for n in range(1, N + 1)) / denom
    return delta

def extract_mfcc(
    audio_path: str,
    n_mfcc: int = 13,
    n_filters: int = 26,
    n_fft: int = 512,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 10.0,
    pre_emph_coef: float = 0.97,
    window: str = "hamming",
    f_min: float = 0.0,
    f_max: float = None,
    include_deltas: bool = True,
) -> dict:
    signal, sr = load_audio(audio_path)

    frame_len = int(sr * frame_duration_ms / 1000)
    hop_len   = int(sr * hop_duration_ms  / 1000)

    emphasized  = pre_emphasis(signal, coef=pre_emph_coef)
    frames      = frame_signal(emphasized, frame_len, hop_len)
    win         = get_window(window, frame_len)
    frames      = frames * win
    pspec       = power_spectrum(frames, n_fft)
    filterbank  = mel_filterbank(n_filters, n_fft, sr, f_min, f_max)
    log_mel     = log_mel_energies(pspec, filterbank)
    mfcc        = apply_dct(log_mel, n_mfcc)

    n_frames = mfcc.shape[0]
    frame_times = (np.arange(n_frames) * hop_len + frame_len // 2) / sr

    result = {
        "mfcc": mfcc,
        "log_mel": log_mel,
        "power_spec": pspec,
        "sr": sr,
        "frame_times": frame_times,
        "filterbank": filterbank,
    }

    if include_deltas:
        delta  = delta_coefficients(mfcc, N=2)
        delta2 = delta_coefficients(delta, N=2)
        result["delta"]  = delta
        result["delta2"] = delta2

    return result

def compute_cepstrum(signal: np.ndarray, sr: int, frame_len: int, hop_len: int,
                     n_fft: int = 512, window: str = "hamming") -> np.ndarray:
    """
    Real cepstrum per frame: IFFT( log |FFT(frame)| )
    Returns shape: (n_frames, n_fft)
    """
    frames  = frame_signal(signal, frame_len, hop_len)
    win     = get_window(window, frame_len)
    frames  = frames * win
    mag     = np.abs(np.fft.fft(frames, n=n_fft))
    log_mag = np.log(mag + 1e-10)
    cepstrum = np.fft.ifft(log_mag, n=n_fft).real
    return cepstrum

def plot_mfcc(result: dict, title: str = "MFCC", save_path: str = None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    t = result["frame_times"]

    im0 = axes[0].imshow(result["mfcc"].T, aspect="auto", origin="lower",
                          extent=[t[0], t[-1], 0, result["mfcc"].shape[1]])
    axes[0].set_title(f"{title} — MFCC coefficients")
    axes[0].set_ylabel("MFCC index")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(result["log_mel"].T, aspect="auto", origin="lower",
                          extent=[t[0], t[-1], 0, result["log_mel"].shape[1]])
    axes[1].set_title("Log-Mel spectrogram")
    axes[1].set_ylabel("Mel filter index")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        10 * np.log10(result["power_spec"].T + 1e-10),
        aspect="auto", origin="lower",
        extent=[t[0], t[-1], 0, result["power_spec"].shape[1]],
    )
    axes[2].set_title("Power spectrum (dB)")
    axes[2].set_ylabel("FFT bin")
    axes[2].set_xlabel("Time (s)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_filterbank(filterbank: np.ndarray, sr: int, n_fft: int,
                    save_path: str = None):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    plt.figure(figsize=(10, 4))
    for f in filterbank:
        plt.plot(freqs, f)
    plt.title("Mel Filterbank")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Manual MFCC Extraction Engine")
    parser.add_argument("--audio",    required=True,         help="Path to input .wav file")
    parser.add_argument("--n_mfcc",   type=int, default=13,  help="Number of MFCC coefficients")
    parser.add_argument("--n_filters",type=int, default=26,  help="Number of Mel filters")
    parser.add_argument("--n_fft",    type=int, default=512, help="FFT size")
    parser.add_argument("--window",   default="hamming",     help="Window: hamming/hanning/rectangular")
    parser.add_argument("--frame_ms", type=float, default=25.0)
    parser.add_argument("--hop_ms",   type=float, default=10.0)
    parser.add_argument("--save_dir", default=".",           help="Directory to save plots")
    args = parser.parse_args()

    print(f"[mfcc_manual] Extracting MFCCs from: {args.audio}")
    result = extract_mfcc(
        args.audio,
        n_mfcc=args.n_mfcc,
        n_filters=args.n_filters,
        n_fft=args.n_fft,
        window=args.window,
        frame_duration_ms=args.frame_ms,
        hop_duration_ms=args.hop_ms,
    )

    print(f"  MFCC shape : {result['mfcc'].shape}")
    print(f"  Δ shape    : {result['delta'].shape}")
    print(f"  ΔΔ shape   : {result['delta2'].shape}")

    import os
    plot_mfcc(result, title=f"MFCC ({args.window} window)",
              save_path=os.path.join(args.save_dir, "mfcc_plot.png"))
    plot_filterbank(result["filterbank"], result["sr"], args.n_fft,
                    save_path=os.path.join(args.save_dir, "filterbank_plot.png"))

    np.save(os.path.join(args.save_dir, "mfcc.npy"), result["mfcc"])
    print("[mfcc_manual] Done. Plots and npy saved.")


if __name__ == "__main__":
    main()
