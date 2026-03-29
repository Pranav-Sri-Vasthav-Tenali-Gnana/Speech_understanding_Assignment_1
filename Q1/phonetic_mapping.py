import argparse
import os
import csv
import json

import numpy as np
import torch
import soundfile as sf

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from dataclasses import dataclass

def load_audio(path: str, target_sr: int = 16000):
    waveform, sr = sf.read(path)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = waveform.astype(np.float32)

    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        waveform = resample_poly(waveform, up, down).astype(np.float32)
        sr = target_sr

    return waveform, sr

def load_segments_csv(path: str) -> list:
    segments = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                "start_s": float(row["start_s"]),
                "end_s":   float(row["end_s"]),
                "label":   row["label"],
            })
    return segments

MODEL_ID = "facebook/wav2vec2-base-960h"


def load_model_and_processor(model_id: str = MODEL_ID):
    print(f"[phonetic_mapping] Loading model: {model_id}")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model     = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return processor, model


@dataclass
class AlignedPhone:
    phone: str
    start_s: float
    end_s: float
    score: float


def get_emission_matrix(waveform: np.ndarray, processor, model, sr: int = 16000) -> tuple:
    inputs = processor(
        waveform, sampling_rate=sr, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    log_probs      = torch.log_softmax(logits[0], dim=-1)
    vocab          = processor.tokenizer.get_vocab()
    idx_to_char    = {v: k for k, v in vocab.items()}
    time_per_frame = len(waveform) / (sr * log_probs.shape[0])

    return log_probs.numpy(), idx_to_char, time_per_frame


def greedy_decode(log_probs: np.ndarray, idx_to_char: dict) -> list:
    pred_ids  = np.argmax(log_probs, axis=-1)
    pad_id    = 0  # Wav2Vec2 uses 0 as blank/pad

    tokens = []
    prev   = None
    for t, idx in enumerate(pred_ids):
        if idx != pad_id and idx != prev:
            tokens.append((idx_to_char.get(idx, "?"), t))
        prev = idx
    return tokens


def force_align(
    waveform: np.ndarray,
    transcript: str,
    processor,
    model,
    sr: int = 16000,
) -> list[AlignedPhone]:
    log_probs, idx_to_char, tpf = get_emission_matrix(waveform, processor, model, sr)
    char_to_idx = {v: k for k, v in idx_to_char.items()}

    transcript_clean = transcript.upper().replace(" ", "|")
    phones = list(transcript_clean)

    T = log_probs.shape[0]
    N = len(phones)

    if N == 0:
        return []
    blank_id = 0
    phone_ids = [char_to_idx.get(p, blank_id) for p in phones]

    ctc_labels = [blank_id]
    for pid in phone_ids:
        ctc_labels.append(pid)
        ctc_labels.append(blank_id)

    S = len(ctc_labels)
    NEG_INF = -1e30

    alpha   = np.full((T, S), NEG_INF)
    backptr = np.zeros((T, S), dtype=int)

    alpha[0, 0] = log_probs[0, ctc_labels[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ctc_labels[1]]

    def log_sum(a, b):
        return np.logaddexp(a, b)

    for t in range(1, T):
        for s in range(S):
            best = NEG_INF
            best_prev = s
            candidates = [s]
            if s > 0:
                candidates.append(s - 1)
            if s > 1 and ctc_labels[s] != blank_id and ctc_labels[s] != ctc_labels[s - 2]:
                candidates.append(s - 2)

            for prev_s in candidates:
                val = alpha[t - 1, prev_s]
                if val > best:
                    best = val
                    best_prev = prev_s

            alpha[t, s]   = best + log_probs[t, ctc_labels[s]]
            backptr[t, s] = best_prev

    last_s = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    path = [last_s]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    aligned_phones = []
    prev_phone_state = -1
    phone_start_frame = 0

    for t, s in enumerate(path):
        label_id = ctc_labels[s]
        if label_id != blank_id:
            if s != prev_phone_state:
                if prev_phone_state != -1 and ctc_labels[prev_phone_state] != blank_id:
                    pass
                phone_start_frame = t
            prev_phone_state = s
        else:
            if prev_phone_state != -1 and ctc_labels[prev_phone_state] != blank_id:
                phone = idx_to_char.get(ctc_labels[prev_phone_state], "?")
                start = phone_start_frame * tpf
                end   = t * tpf
                score = float(np.exp(alpha[t, prev_phone_state]))
                aligned_phones.append(AlignedPhone(phone, start, end, score))
                prev_phone_state = -1

    if prev_phone_state != -1 and ctc_labels[prev_phone_state] != blank_id:
        phone = idx_to_char.get(ctc_labels[prev_phone_state], "?")
        aligned_phones.append(AlignedPhone(phone, phone_start_frame * tpf,
                                           T * tpf, 0.0))

    return aligned_phones

def transcribe(waveform: np.ndarray, processor, model, sr: int = 16000) -> str:
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids   = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(pred_ids)[0]
    return transcript

def extract_boundary_times(segments: list, key: str = "start_s") -> np.ndarray:
    return np.array([s[key] for s in segments])


def match_boundaries(manual_bounds: np.ndarray,
                     model_bounds: np.ndarray,
                     max_gap_s: float = 0.3) -> tuple:
    matched_m, matched_a = [], []
    used = set()
    for mb in manual_bounds:
        dists = np.abs(model_bounds - mb)
        best  = int(np.argmin(dists))
        if dists[best] <= max_gap_s and best not in used:
            matched_m.append(mb)
            matched_a.append(model_bounds[best])
            used.add(best)
    return np.array(matched_m), np.array(matched_a)


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))
VOICED_PHONES = set("AEIOUYBDGVWZJMN")   # vowels + voiced consonants
UNVOICED_PHONES = set("PTKFSHX")


def phone_type(phone: str) -> str:
    p = phone.upper().strip()
    if not p or p == "|":
        return "silence"
    if p[0] in VOICED_PHONES:
        return "voiced"
    if p[0] in UNVOICED_PHONES:
        return "unvoiced"
    return "unknown"


def map_phones_to_segments(aligned_phones: list[AlignedPhone],
                           manual_segments: list) -> list:
    mapping = []
    for ap in aligned_phones:
        mid = (ap.start_s + ap.end_s) / 2.0
        matched_seg = None
        for seg in manual_segments:
            if seg["start_s"] <= mid < seg["end_s"]:
                matched_seg = seg
                break
        mapping.append({
            "phone":      ap.phone,
            "phone_type": phone_type(ap.phone),
            "start_s":    ap.start_s,
            "end_s":      ap.end_s,
            "manual_label": matched_seg["label"] if matched_seg else "none",
        })
    return mapping

def plot_alignment(waveform: np.ndarray, sr: int,
                   manual_segs: list,
                   aligned_phones: list[AlignedPhone],
                   save_path: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t = np.arange(len(waveform)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, waveform, color="steelblue", linewidth=0.6, alpha=0.7)
    for seg in manual_segs:
        color = "#2ecc71" if seg["label"] == "voiced" else "#e74c3c"
        ax.axvspan(seg["start_s"], seg["end_s"], alpha=0.25, color=color)
    ax.set_ylabel("Amplitude")
    ax.set_title("Manual Voiced/Unvoiced Segments")
    ax.set_xlim([t[0], t[-1]])

    ax = axes[1]
    ax.plot(t, waveform, color="gray", linewidth=0.5, alpha=0.4)
    colors_map = {"voiced": "#2980b9", "unvoiced": "#c0392b",
                  "silence": "#bdc3c7", "unknown": "#7f8c8d"}
    for ap in aligned_phones:
        c = colors_map.get(phone_type(ap.phone), "#7f8c8d")
        ax.axvspan(ap.start_s, ap.end_s, alpha=0.35, color=c)
        mid = (ap.start_s + ap.end_s) / 2.0
        ax.text(mid, 0.8 * waveform.max(), ap.phone,
                ha="center", va="bottom", fontsize=7, color="black")

    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_title("Wav2Vec2 Phone Alignment")
    ax.set_xlim([t[0], t[-1]])

    patches = [mpatches.Patch(color=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[phonetic_mapping] Alignment plot saved → {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Phonetic Mapping & RMSE")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--segments", required=True, help="Path to segments.csv (from voiced_unvoiced.py)")
    parser.add_argument("--save_dir", default=".",   help="Output directory")
    parser.add_argument("--model_id", default=MODEL_ID)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    waveform, sr    = load_audio(args.audio)
    manual_segments = load_segments_csv(args.segments)
    processor, model = load_model_and_processor(args.model_id)

    transcript = transcribe(waveform, processor, model, sr)
    print(f"[phonetic_mapping] Transcript: {transcript!r}")

    print("[phonetic_mapping] Running forced alignment …")
    aligned_phones = force_align(waveform, transcript, processor, model, sr)
    print(f"[phonetic_mapping] Aligned {len(aligned_phones)} phones.")

    phone_mapping = map_phones_to_segments(aligned_phones, manual_segments)

    manual_starts = np.array([s["start_s"] for s in manual_segments[1:]])   # skip first
    model_starts  = np.array([ap.start_s   for ap in aligned_phones])

    m_matched, a_matched = match_boundaries(manual_starts, model_starts)
    rmse = compute_rmse(m_matched, a_matched)
    print(f"\n[phonetic_mapping] Boundary RMSE: {rmse*1000:.2f} ms  "
          f"(over {len(m_matched)} matched pairs)")

    mapping_path = os.path.join(args.save_dir, "phone_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phone", "phone_type", "start_s",
                                               "end_s", "manual_label"])
        writer.writeheader()
        writer.writerows(phone_mapping)
    print(f"[phonetic_mapping] Phone mapping → {mapping_path}")

    summary = {
        "transcript":        transcript,
        "n_phones":          len(aligned_phones),
        "n_manual_segments": len(manual_segments),
        "matched_pairs":     int(len(m_matched)),
        "rmse_seconds":      float(rmse),
        "rmse_ms":           float(rmse * 1000) if not np.isnan(rmse) else None,
    }
    summary_path = os.path.join(args.save_dir, "rmse_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phonetic_mapping] RMSE summary → {summary_path}")

    # Plot
    plot_alignment(waveform, sr, manual_segments, aligned_phones,
                   save_path=os.path.join(args.save_dir, "phone_alignment.png"))

    print("\n── RMSE Table ─────────────────────────────────────────")
    print(f"{'Metric':<35} {'Value':>12}")
    print("-" * 50)
    print(f"{'Number of manual segments':<35} {len(manual_segments):>12}")
    print(f"{'Number of model phones':<35} {len(aligned_phones):>12}")
    print(f"{'Matched boundary pairs':<35} {len(m_matched):>12}")
    print(f"{'Boundary RMSE (ms)':<35} {rmse*1000:>12.2f}")
    print("─" * 50)

    print("[phonetic_mapping] Done.")


if __name__ == "__main__":
    main()
