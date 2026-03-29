import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from privacymodule import PrivacyPreservingModule


def load_audio(path, target_sr=16000):
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def compute_snr_estimate(waveform, frame_len=400, hop_len=160, percentile=10):
    signal = waveform.squeeze(0).numpy()
    n_frames = (len(signal) - frame_len) // hop_len + 1
    if n_frames < 1:
        return 0.0
    energies = []
    for i in range(n_frames):
        frame = signal[i * hop_len: i * hop_len + frame_len]
        energies.append(np.mean(frame ** 2))
    energies = np.array(energies)
    noise_floor = np.percentile(energies, percentile)
    signal_power = np.mean(energies)
    snr = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
    return float(snr)


def compute_spectral_flatness(waveform, n_fft=512, hop_length=128):
    spec = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    ).abs()
    spec = spec + 1e-10
    log_mean = spec.log().mean(dim=0)
    arith_mean = spec.mean(dim=0).log()
    flatness = (log_mean - arith_mean).mean().item()
    return float(flatness)


def compute_zcr(waveform):
    signal = waveform.squeeze(0).numpy()
    zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2
    return float(zcr)


def compute_energy_entropy(waveform, frame_len=400, hop_len=160, n_subframes=4):
    signal = waveform.squeeze(0).numpy()
    n_frames = (len(signal) - frame_len) // hop_len + 1
    entropies = []
    for i in range(n_frames):
        frame = signal[i * hop_len: i * hop_len + frame_len]
        sub_len = frame_len // n_subframes
        sub_energies = np.array([
            np.sum(frame[j * sub_len:(j + 1) * sub_len] ** 2)
            for j in range(n_subframes)
        ])
        total = sub_energies.sum()
        if total < 1e-10:
            continue
        probs = sub_energies / total + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)
    return float(np.mean(entropies)) if entropies else 0.0


def proxy_dnsmos(waveform, sr=16000):
    snr = compute_snr_estimate(waveform)
    flatness = compute_spectral_flatness(waveform)
    zcr = compute_zcr(waveform)
    entropy = compute_energy_entropy(waveform)

    snr_score = min(max((snr - 5) / 30, 0), 1)
    flatness_score = min(max((-flatness - 1) / 4, 0), 1)
    zcr_score = 1.0 - min(zcr / 0.3, 1)
    entropy_score = min(entropy / np.log(4), 1)

    overall = 1.0 + 3.0 * (0.4 * snr_score + 0.3 * flatness_score + 0.2 * entropy_score + 0.1 * zcr_score)

    return {
        "proxy_dnsmos": round(float(overall), 3),
        "snr_estimate_db": round(snr, 2),
        "spectral_flatness": round(flatness, 4),
        "zero_crossing_rate": round(zcr, 4),
        "energy_entropy": round(entropy, 4),
    }


def scan_audio_pairs(examples_dir, tag):
    examples_dir = Path(examples_dir)
    pairs = []
    for orig in sorted(examples_dir.glob(f"{tag}_*_original.wav")):
        stem = orig.stem.replace("_original", "")
        trans = examples_dir / f"{stem}_transformed.wav"
        if trans.exists():
            pairs.append((orig, trans))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_dir", default="examples")
    parser.add_argument("--out", default="results/dnsmos_results.json")
    args = parser.parse_args()

    pairs_mf = scan_audio_pairs(args.examples_dir, "male_to_female")
    pairs_fm = scan_audio_pairs(args.examples_dir, "female_to_male")
    all_pairs = [("M->F", p) for p in pairs_mf] + [("F->M", p) for p in pairs_fm]

    if not all_pairs:
        print(f"No audio pairs found in {args.examples_dir}. Run pp_demo.py first.")
        return

    results = []
    print(f"{'Tag':<8} {'File':<40} {'Type':<12} {'DNSMOS':>8} {'SNR (dB)':>10} {'Flatness':>10}")
    print("-" * 90)

    for tag, (orig_path, trans_path) in all_pairs:
        orig_wav, sr = load_audio(orig_path)
        trans_wav, _ = load_audio(trans_path)

        orig_scores = proxy_dnsmos(orig_wav, sr)
        trans_scores = proxy_dnsmos(trans_wav, sr)

        print(f"{tag:<8} {orig_path.name:<40} {'Original':<12} "
              f"{orig_scores['proxy_dnsmos']:>8.3f} {orig_scores['snr_estimate_db']:>10.2f} "
              f"{orig_scores['spectral_flatness']:>10.4f}")
        print(f"{tag:<8} {trans_path.name:<40} {'Transformed':<12} "
              f"{trans_scores['proxy_dnsmos']:>8.3f} {trans_scores['snr_estimate_db']:>10.2f} "
              f"{trans_scores['spectral_flatness']:>10.4f}")
        print()

        results.append({
            "tag": tag,
            "file": orig_path.name,
            "original": orig_scores,
            "transformed": trans_scores,
            "dnsmos_degradation": round(orig_scores["proxy_dnsmos"] - trans_scores["proxy_dnsmos"], 3),
        })

    if results:
        avg_degradation = np.mean([r["dnsmos_degradation"] for r in results])
        print(f"Average DNSMOS degradation after transformation: {avg_degradation:.3f}")
        print("(< 0.5 is acceptable; > 1.0 suggests significant artifacts)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
