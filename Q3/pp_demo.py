import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from privacymodule import PrivacyPreservingModule, load_audio, save_audio


SPEAKERS_TXT = "../Q2/data/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
LIBRISPEECH_ROOT = "../Q2/data/LibriSpeech/LibriSpeech"
TARGET_SUBSETS = ["train-clean-100"]


def parse_gender_map(speakers_txt):
    gender_map = {}
    with open(speakers_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            try:
                sid = int(parts[0])
            except ValueError:
                continue
            gender = parts[1].upper()
            gender_map[sid] = gender
    return gender_map


def find_flac_files(root, subsets, gender_map, gender, n=3):
    root = Path(root)
    found = []
    for subset in subsets:
        subset_path = root / subset
        if not subset_path.exists():
            continue
        for flac in subset_path.rglob("*.flac"):
            parts = flac.stem.split("-")
            if len(parts) != 3:
                continue
            sid = int(parts[0])
            if gender_map.get(sid, "?") == gender:
                found.append((flac, sid))
            if len(found) >= n:
                return found
    return found


def plot_waveform_comparison(orig, transformed, sr, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    t_orig = np.arange(orig.shape[-1]) / sr
    t_trans = np.arange(transformed.shape[-1]) / sr

    axes[0, 0].plot(t_orig, orig.squeeze(0).numpy(), linewidth=0.5, color="#4C72B0")
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t_trans, transformed.squeeze(0).numpy(), linewidth=0.5, color="#DD8452")
    axes[0, 1].set_title("Transformed Waveform")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")

    n_fft = 512
    hop = 128

    def get_spectrogram(w):
        spec = torch.stft(w.squeeze(0), n_fft=n_fft, hop_length=hop, return_complex=True)
        return spec.abs().log1p().numpy()

    spec_orig = get_spectrogram(orig)
    spec_trans = get_spectrogram(transformed)

    axes[1, 0].imshow(
        spec_orig, origin="lower", aspect="auto", cmap="magma",
        extent=[0, orig.shape[-1] / sr, 0, sr / 2 / 1000]
    )
    axes[1, 0].set_title("Original Spectrogram")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (kHz)")

    axes[1, 1].imshow(
        spec_trans, origin="lower", aspect="auto", cmap="magma",
        extent=[0, transformed.shape[-1] / sr, 0, sr / 2 / 1000]
    )
    axes[1, 1].set_title("Transformed Spectrogram")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (kHz)")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_snr(original, transformed):
    noise = original - transformed
    signal_power = original.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers_txt", default=SPEAKERS_TXT)
    parser.add_argument("--librispeech_root", default=LIBRISPEECH_ROOT)
    parser.add_argument("--subsets", nargs="+", default=TARGET_SUBSETS)
    parser.add_argument("--examples_dir", default="examples")
    parser.add_argument("--n_examples", type=int, default=3)
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    gender_map = parse_gender_map(args.speakers_txt)
    module = PrivacyPreservingModule(sample_rate=16000)

    transformations = [
        ("M", "F", "male_to_female"),
        ("F", "M", "female_to_male"),
    ]

    results = []

    for src_gender, tgt_gender, tag in transformations:
        files = find_flac_files(
            args.librispeech_root, args.subsets, gender_map, src_gender, n=args.n_examples
        )
        if not files:
            print(f"No {src_gender} speakers found in {args.subsets}")
            continue

        print(f"\n{src_gender} -> {tgt_gender} ({len(files)} examples)")

        for i, (flac_path, sid) in enumerate(files):
            waveform, sr = load_audio(flac_path)

            if waveform.shape[-1] > sr * 5:
                waveform = waveform[..., :sr * 5]

            transformed = module.transform(waveform, src_gender, tgt_gender)
            snr = compute_snr(waveform, transformed)

            orig_out = examples_dir / f"{tag}_example{i+1}_original.wav"
            trans_out = examples_dir / f"{tag}_example{i+1}_transformed.wav"
            save_audio(waveform, orig_out, sr)
            save_audio(transformed, trans_out, sr)

            plot_waveform_comparison(
                waveform, transformed, sr,
                title=f"Speaker {sid}: {src_gender} → {tgt_gender} (example {i+1})",
                save_path=plots_dir / f"{tag}_example{i+1}.png",
            )

            print(f"  [{i+1}] Speaker {sid} | SNR of change: {snr:.2f} dB")
            results.append({
                "speaker_id": sid,
                "src_gender": src_gender,
                "tgt_gender": tgt_gender,
                "snr_db": round(snr, 3),
                "original": str(orig_out),
                "transformed": str(trans_out),
            })

    import json
    with open(Path("results") / "pp_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExamples saved to {examples_dir}/")
    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
