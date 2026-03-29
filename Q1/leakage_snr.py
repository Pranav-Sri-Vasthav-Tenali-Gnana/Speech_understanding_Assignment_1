import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import csv
import os

def get_window(name: str, length: int) -> np.ndarray:
    name = name.lower()
    if name == "rectangular":
        return np.ones(length)
    elif name == "hamming":
        return np.hamming(length)
    elif name in ("hanning", "hann"):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))
    raise ValueError(f"Unknown window: {name}")


WINDOWS = ["rectangular", "hamming", "hanning"]


def spectral_leakage_metrics(signal: np.ndarray, window_name: str, n_fft: int = 2048) -> dict:

    win = get_window(window_name, len(signal))
    windowed = signal * win
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power    = np.abs(spectrum) ** 2

    total_energy = power.sum() + 1e-10

    peak_bin = np.argmax(power)
    lobe_half_width = max(3, int(0.01 * n_fft))   # at least 3 bins
    lobe_start = max(0, peak_bin - lobe_half_width)
    lobe_end   = min(len(power), peak_bin + lobe_half_width + 1)

    main_lobe_energy = power[lobe_start:lobe_end].sum()
    side_lobe_energy = total_energy - main_lobe_energy

    leakage_ratio = side_lobe_energy / total_energy

    side_power = power.copy()
    side_power[lobe_start:lobe_end] = 0.0
    if side_power.max() > 0 and power[peak_bin] > 0:
        side_lobe_dB = 10 * np.log10(side_power.max() / power[peak_bin])
    else:
        side_lobe_dB = -np.inf

    return {
        "window": window_name,
        "leakage_ratio": leakage_ratio,
        "side_lobe_dB": side_lobe_dB,
        "spectrum_db": 10 * np.log10(power + 1e-10),
        "freqs_idx": np.arange(len(power)),
    }

def estimate_snr(signal: np.ndarray, window_name: str, n_fft: int = 2048,
                 noise_percentile: float = 10.0) -> float:

    win = get_window(window_name, len(signal))
    windowed = signal * win
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power    = np.abs(spectrum) ** 2

    # Noise floor: average of lowest N% of bins
    threshold = np.percentile(power, noise_percentile)
    noise_mask  = power <= threshold
    signal_mask = power > threshold

    noise_power  = power[noise_mask].mean() if noise_mask.any() else 1e-10
    signal_power = power[signal_mask].mean() if signal_mask.any() else 1e-10

    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr_db

def compare_windows(segment: np.ndarray, sr: int, n_fft: int = 2048,
                    save_dir: str = ".") -> list:
    results = []
    for wname in WINDOWS:
        metrics = spectral_leakage_metrics(segment, wname, n_fft)
        snr     = estimate_snr(segment, wname, n_fft)
        metrics["snr_db"] = snr
        results.append(metrics)
    return results


def print_table(results: list):
    header = f"{'Window':<15} {'Leakage Ratio':>15} {'Side-Lobe (dB)':>16} {'SNR (dB)':>10}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['window']:<15} {r['leakage_ratio']:>15.6f} {r['side_lobe_dB']:>16.2f} {r['snr_db']:>10.2f}")
    print(sep + "\n")


def save_table_csv(results: list, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "leakage_ratio", "side_lobe_dB", "snr_db"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["window", "leakage_ratio", "side_lobe_dB", "snr_db"]})
    print(f"[leakage_snr] Table saved → {path}")

def plot_spectra(results: list, sr: int, n_fft: int, save_path: str = None):
    colors = {"rectangular": "#e74c3c", "hamming": "#2980b9", "hanning": "#27ae60"}
    fig, ax = plt.subplots(figsize=(12, 5))

    for r in results:
        freqs = r["freqs_idx"] * sr / n_fft
        ax.plot(freqs, r["spectrum_db"], label=r["window"].capitalize(),
                color=colors[r["window"]], alpha=0.85, linewidth=1.2)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Spectral Leakage Comparison — Three Windows")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[leakage_snr] Spectra plot saved → {save_path}")


def plot_window_shapes(save_path: str = None):
    N = 512
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = {"rectangular": "-", "hamming": "--", "hanning": "-."}
    colors = {"rectangular": "#e74c3c", "hamming": "#2980b9", "hanning": "#27ae60"}
    for wname in WINDOWS:
        w = get_window(wname, N)
        ax.plot(w, label=wname.capitalize(), linestyle=styles[wname],
                color=colors[wname], linewidth=1.8)
    ax.set_title("Window Functions (N=512)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_snr_bar(results: list, save_path: str = None):
    names = [r["window"].capitalize() for r in results]
    snrs  = [r["snr_db"] for r in results]
    colors = ["#e74c3c", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, snrs, color=colors, edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%.2f dB", padding=3)
    ax.set_ylabel("Estimated SNR (dB)")
    ax.set_title("SNR Comparison Across Window Functions")
    ax.set_ylim(0, max(snrs) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[leakage_snr] SNR bar chart saved → {save_path}")


def plot_leakage_bar(results: list, save_path: str = None):
    names    = [r["window"].capitalize() for r in results]
    leakages = [r["leakage_ratio"] * 100 for r in results]   # percent
    colors   = ["#e74c3c", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, leakages, color=colors, edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%.2f %%", padding=3)
    ax.set_ylabel("Leakage Ratio (%)")
    ax.set_title("Spectral Leakage Ratio Across Window Functions")
    ax.set_ylim(0, max(leakages) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[leakage_snr] Leakage bar chart saved → {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Spectral Leakage & SNR Analysis")
    parser.add_argument("--audio",           required=True,         help="Path to .wav file")
    parser.add_argument("--segment_start",   type=float, default=0.0, help="Segment start (s)")
    parser.add_argument("--segment_end",     type=float, default=1.0, help="Segment end (s)")
    parser.add_argument("--n_fft",           type=int,   default=2048)
    parser.add_argument("--save_dir",        default=".",           help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    signal, sr = sf.read(args.audio, always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)
    start_sample = int(args.segment_start * sr)
    end_sample   = int(args.segment_end   * sr)
    segment = signal[start_sample:end_sample]
    print(f"[leakage_snr] Segment: {args.segment_start:.2f}s → {args.segment_end:.2f}s "
          f"({len(segment)} samples @ {sr} Hz)")

    results = compare_windows(segment, sr, n_fft=args.n_fft, save_dir=args.save_dir)

    print_table(results)
    save_table_csv(results, os.path.join(args.save_dir, "leakage_snr_table.csv"))

    plot_window_shapes(save_path=os.path.join(args.save_dir, "window_shapes.png"))
    plot_spectra(results, sr, args.n_fft,
                 save_path=os.path.join(args.save_dir, "spectra_comparison.png"))
    plot_snr_bar(results,
                 save_path=os.path.join(args.save_dir, "snr_comparison.png"))
    plot_leakage_bar(results,
                     save_path=os.path.join(args.save_dir, "leakage_comparison.png"))

    print("[leakage_snr] All done.")


if __name__ == "__main__":
    main()
