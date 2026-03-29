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


def extract_mfcc_features(waveform, sr):
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
    )
    mfcc = mfcc_t(waveform).squeeze(0)
    mean = mfcc.mean(dim=1)
    std = mfcc.std(dim=1)
    return torch.cat([mean, std]).numpy()


def compute_fad_proxy(original_features, transformed_features):
    mu1 = np.mean(original_features, axis=0)
    mu2 = np.mean(transformed_features, axis=0)
    sigma1 = np.cov(original_features, rowvar=False)
    sigma2 = np.cov(transformed_features, rowvar=False)

    diff = mu1 - mu2
    mean_dist = np.dot(diff, diff)

    cov_mean = (sigma1 + sigma2) / 2.0
    trace_term = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(
        sqrtm_approx(sigma1 @ sigma2)
    )

    fad = mean_dist + trace_term
    return float(fad)


def sqrtm_approx(M):
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def scan_audio_files(directory, extensions=(".flac", ".wav")):
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True, help="Directory with original audio files")
    parser.add_argument("--transformed_dir", default=None, help="Directory with transformed audio (optional)")
    parser.add_argument("--src_gender", default="M")
    parser.add_argument("--tgt_gender", default="F")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--out", default="results/fad_results.json")
    args = parser.parse_args()

    module = PrivacyPreservingModule(sample_rate=16000)

    orig_files = scan_audio_files(args.original_dir)[:args.max_files]
    if not orig_files:
        print(f"No audio files found in {args.original_dir}")
        return

    print(f"Processing {len(orig_files)} files...")

    orig_features = []
    trans_features = []

    for path in orig_files:
        try:
            waveform, sr = load_audio(path)
            if waveform.shape[-1] < sr * 0.5:
                continue

            orig_feat = extract_mfcc_features(waveform, sr)
            orig_features.append(orig_feat)

            if args.transformed_dir:
                trans_path = Path(args.transformed_dir) / path.name
                if trans_path.exists():
                    trans_wav, _ = load_audio(trans_path)
                else:
                    trans_wav = module.transform(waveform, args.src_gender, args.tgt_gender)
            else:
                trans_wav = module.transform(waveform, args.src_gender, args.tgt_gender)

            trans_feat = extract_mfcc_features(trans_wav, sr)
            trans_features.append(trans_feat)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
            continue

    if len(orig_features) < 2:
        print("Not enough files for FAD computation (need at least 2)")
        return

    orig_features = np.array(orig_features)
    trans_features = np.array(trans_features)

    fad = compute_fad_proxy(orig_features, trans_features)

    mean_diff = np.mean(np.abs(orig_features - trans_features))
    cosine_sims = []
    for o, t in zip(orig_features, trans_features):
        cos = np.dot(o, t) / (np.linalg.norm(o) * np.linalg.norm(t) + 1e-8)
        cosine_sims.append(cos)
    mean_cosine = float(np.mean(cosine_sims))

    result = {
        "fad_proxy": round(fad, 4),
        "mean_feature_diff": round(float(mean_diff), 4),
        "mean_cosine_similarity": round(mean_cosine, 4),
        "n_files": len(orig_features),
        "transformation": f"{args.src_gender} -> {args.tgt_gender}",
    }

    print(f"\n=== FAD Proxy Results ({args.src_gender} -> {args.tgt_gender}) ===")
    print(f"FAD Proxy:              {result['fad_proxy']:.4f}")
    print(f"Mean Feature Diff:      {result['mean_feature_diff']:.4f}")
    print(f"Mean Cosine Similarity: {result['mean_cosine_similarity']:.4f}")
    print(f"  (1.0 = identical, lower = more distortion)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
