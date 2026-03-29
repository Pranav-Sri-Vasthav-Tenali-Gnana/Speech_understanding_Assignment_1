import argparse
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import medfilt

def load_audio(path):
    signal, sr = sf.read(path, always_2d=False)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32), sr


def get_window(name, length):
    name = name.lower()
    if name == "hamming":
        return np.hamming(length)
    elif name in ("hanning", "hann"):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))
    return np.ones(length)


def frame_signal(signal, frame_len, hop_len):
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    idx = (np.tile(np.arange(frame_len), (n_frames, 1))
           + np.tile(np.arange(n_frames) * hop_len, (frame_len, 1)).T)
    return signal[idx]

def cepstral_voiced_score(frames: np.ndarray, sr: int,
                           pitch_min_hz: float = 60.0,
                           pitch_max_hz: float = 400.0,
                           n_fft: int = 512) -> np.ndarray:
    win = get_window("hamming", frames.shape[1])
    windowed = frames * win
    mag_spec  = np.abs(np.fft.fft(windowed, n=n_fft))
    log_spec  = np.log(mag_spec + 1e-10)
    cepstrum  = np.fft.ifft(log_spec, n=n_fft).real

    q_min = int(sr / pitch_max_hz)
    q_max = int(sr / pitch_min_hz)
    q_min = max(1, q_min)
    q_max = min(n_fft // 2, q_max)

    high_q_start = int(0.002 * sr)   # 2 ms in samples
    high_q_start = max(1, high_q_start)

    scores = np.zeros(len(frames))
    for i, cep in enumerate(cepstrum):
        high_region = np.abs(cep[high_q_start: n_fft // 2])
        pitch_region = np.abs(cep[q_min:q_max])

        if high_region.sum() < 1e-12:
            scores[i] = 0.0
        else:
            scores[i] = pitch_region.max() / (high_region.max() + 1e-10)

    return scores

def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = (np.diff(signs, axis=1) != 0).sum(axis=1).astype(float)
    crossings /= frames.shape[1]
    # Normalise
    if crossings.max() > crossings.min():
        crossings = (crossings - crossings.min()) / (crossings.max() - crossings.min())
    return crossings

def short_term_energy(frames: np.ndarray) -> np.ndarray:
    energy = (frames ** 2).mean(axis=1)
    if energy.max() > energy.min():
        energy = (energy - energy.min()) / (energy.max() - energy.min())
    return energy

def classify_frames(
    cep_score: np.ndarray,
    ste: np.ndarray,
    zcr: np.ndarray,
    w_cep: float = 0.5,
    w_ste: float = 0.3,
    w_zcr: float = 0.2,
    threshold: float = 0.45,
    median_filter_len: int = 5,
) -> np.ndarray:
    voiced_prob = w_cep * cep_score + w_ste * ste + w_zcr * (1.0 - zcr)

    # Smooth with median filter
    if median_filter_len > 1:
        voiced_prob = medfilt(voiced_prob, kernel_size=median_filter_len)

    return (voiced_prob >= threshold).astype(int), voiced_prob

def get_boundaries(labels: np.ndarray, frame_times: np.ndarray) -> list:
    segments = []
    label_names = {1: "voiced", 0: "unvoiced"}
    if len(labels) == 0:
        return segments

    current_label = labels[0]
    current_start = frame_times[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((current_start, frame_times[i], label_names[current_label]))
            current_label = labels[i]
            current_start = frame_times[i]

    segments.append((current_start, frame_times[-1], label_names[current_label]))
    return segments

def detect_boundaries(
    audio_path: str,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int = 512,
    pitch_min_hz: float = 60.0,
    pitch_max_hz: float = 400.0,
    threshold: float = 0.45,
    median_filter_len: int = 5,
) -> dict:
    signal, sr = load_audio(audio_path)
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms  / 1000)

    frames = frame_signal(signal, frame_len, hop_len)
    n_frames = len(frames)
    frame_times = (np.arange(n_frames) * hop_len + frame_len // 2) / sr

    cep_score = cepstral_voiced_score(frames, sr, pitch_min_hz, pitch_max_hz, n_fft)
    ste       = short_term_energy(frames)
    zcr       = zero_crossing_rate(frames)

    labels, voiced_prob = classify_frames(cep_score, ste, zcr,
                                           threshold=threshold,
                                           median_filter_len=median_filter_len)
    segments = get_boundaries(labels, frame_times)

    return {
        "signal": signal,
        "sr": sr,
        "frame_times": frame_times,
        "cep_score": cep_score,
        "ste": ste,
        "zcr": zcr,
        "voiced_prob": voiced_prob,
        "labels": labels,
        "segments": segments,
    }

def plot_results(result: dict, title: str = "", save_path: str = None):
    signal      = result["signal"]
    sr          = result["sr"]
    frame_times = result["frame_times"]
    labels      = result["labels"]
    voiced_prob = result["voiced_prob"]
    segments    = result["segments"]

    t_signal = np.arange(len(signal)) / sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    ax = axes[0]
    ax.plot(t_signal, signal, color="steelblue", linewidth=0.6, alpha=0.8)
    for (start, end, lab) in segments:
        color = "#2ecc71" if lab == "voiced" else "#e74c3c"
        ax.axvspan(start, end, alpha=0.25, color=color)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform with Voiced/Unvoiced Regions  {title}")
    voiced_patch   = mpatches.Patch(color="#2ecc71", alpha=0.5, label="Voiced")
    unvoiced_patch = mpatches.Patch(color="#e74c3c", alpha=0.5, label="Unvoiced")
    ax.legend(handles=[voiced_patch, unvoiced_patch], loc="upper right")
    ax.set_xlim([t_signal[0], t_signal[-1]])
    ax = axes[1]
    ax.plot(frame_times, voiced_prob, color="#8e44ad", linewidth=1.2)
    ax.axhline(y=0.45, color="gray", linestyle="--", linewidth=0.8, label="Threshold")
    ax.fill_between(frame_times, voiced_prob, 0.45,
                    where=voiced_prob >= 0.45, alpha=0.3, color="#2ecc71")
    ax.fill_between(frame_times, voiced_prob, 0.45,
                    where=voiced_prob < 0.45, alpha=0.3, color="#e74c3c")
    ax.set_ylabel("Voiced Prob.")
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper right")
    ax.set_xlim([frame_times[0], frame_times[-1]])

    ax = axes[2]
    ax.plot(frame_times, result["cep_score"], label="Cep. Score", color="#2980b9")
    ax.plot(frame_times, result["ste"],       label="STE",         color="#e67e22")
    ax.plot(frame_times, 1 - result["zcr"],   label="1 - ZCR",     color="#16a085")
    ax.set_ylabel("Feature value")
    ax.legend(loc="upper right")
    ax.set_xlim([frame_times[0], frame_times[-1]])
    ax = axes[3]
    ax.step(frame_times, labels, color="#2c3e50", linewidth=1.2, where="post")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unvoiced", "Voiced"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decision")
    ax.set_xlim([frame_times[0], frame_times[-1]])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[voiced_unvoiced] Plot saved → {save_path}")


def save_segments_csv(segments: list, path: str):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "label"])
        for (start, end, label) in segments:
            writer.writerow([f"{start:.4f}", f"{end:.4f}", label])
    print(f"[voiced_unvoiced] Segments saved → {path}")

def main():
    parser = argparse.ArgumentParser(description="Voiced/Unvoiced Boundary Detection")
    parser.add_argument("--audio",      required=True)
    parser.add_argument("--frame_ms",   type=float, default=25.0)
    parser.add_argument("--hop_ms",     type=float, default=10.0)
    parser.add_argument("--n_fft",      type=int,   default=512)
    parser.add_argument("--pitch_min",  type=float, default=60.0,  help="Min pitch Hz")
    parser.add_argument("--pitch_max",  type=float, default=400.0, help="Max pitch Hz")
    parser.add_argument("--threshold",  type=float, default=0.45,  help="Voiced prob threshold")
    parser.add_argument("--save_dir",   default=".",               help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[voiced_unvoiced] Processing: {args.audio}")
    result = detect_boundaries(
        args.audio,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        pitch_min_hz=args.pitch_min,
        pitch_max_hz=args.pitch_max,
        threshold=args.threshold,
    )
    segs = result["segments"]
    total_voiced   = sum(e - s for s, e, l in segs if l == "voiced")
    total_unvoiced = sum(e - s for s, e, l in segs if l == "unvoiced")
    total_dur = result["signal"].shape[0] / result["sr"]
    print(f"  Total duration : {total_dur:.2f} s")
    print(f"  Voiced         : {total_voiced:.2f} s  ({100*total_voiced/total_dur:.1f}%)")
    print(f"  Unvoiced       : {total_unvoiced:.2f} s  ({100*total_unvoiced/total_dur:.1f}%)")
    print(f"  Segments found : {len(segs)}")

    plot_results(result,
                 title=f"— {os.path.basename(args.audio)}",
                 save_path=os.path.join(args.save_dir, "voiced_unvoiced.png"))

    save_segments_csv(segs, os.path.join(args.save_dir, "segments.csv"))
    np.save(os.path.join(args.save_dir, "frame_labels.npy"), result["labels"])

    print("[voiced_unvoiced] Done.")


if __name__ == "__main__":
    main()
