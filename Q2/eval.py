import os
import json
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE

from models import DisentanglementModel
from dataset import build_verification_trials


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path, emb_dim, latent_dim, env_hidden, num_speakers):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DisentanglementModel(emb_dim, latent_dim, env_hidden, num_speakers)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["speaker_to_idx"]


def cosine_score(emb1, emb2):
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fnr, thresholds)(eer)
    return float(eer) * 100, float(thresh)


def compute_min_dcf(labels, scores, p_target=0.05, c_miss=1.0, c_fa=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = dcf.min()
    return float(min_dcf)


def score_trials(trials, model, use_spk_only=True):
    scores = []
    labels = []
    for path1, path2, label in trials:
        emb1 = torch.load(path1, weights_only=True).float()
        emb2 = torch.load(path2, weights_only=True).float()

        if model is not None:
            with torch.no_grad():
                e_spk1, _ = model.encode(emb1.unsqueeze(0))
                e_spk2, _ = model.encode(emb2.unsqueeze(0))
            score = cosine_score(e_spk1.squeeze(0), e_spk2.squeeze(0))
        else:
            score = cosine_score(emb1, emb2)

        scores.append(score)
        labels.append(label)

    return np.array(labels), np.array(scores)


def plot_det_curve(results, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (labels, scores) in results.items():
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        ax.plot(fpr * 100, fnr * 100, label=name, linewidth=2)

    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title("DET Curve")
    ax.legend()
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_distributions(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (labels, scores)) in zip(axes, results.items()):
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        ax.hist(neg_scores, bins=60, alpha=0.6, label="Different speaker", density=True, color="red")
        ax.hist(pos_scores, bins=60, alpha=0.6, label="Same speaker", density=True, color="blue")
        ax.set_title(name)
        ax.set_xlabel("Cosine Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history_paths, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    keys = ["total", "spk", "recons", "env", "adv", "corr"]
    titles = ["Total Loss", "Speaker Loss", "Reconstruction Loss", "Env Triplet Loss", "Adversarial Loss", "Correlation Loss"]

    for path_info in history_paths:
        name, path = path_info
        with open(path) as f:
            hist = json.load(f)
        epochs = [h["epoch"] for h in hist]
        for ax, key, title in zip(axes, keys, titles):
            vals = [h.get(key, 0) for h in hist]
            ax.plot(epochs, vals, label=name, linewidth=2)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def collect_embeddings_for_tsne(embedding_dir, subset, model, max_speakers=20, max_per_speaker=10):
    meta_path = Path(embedding_dir) / f"{subset}_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    spk_files = defaultdict(list)
    for item in metadata:
        spk_files[item["speaker_id"]].append(
            Path(embedding_dir) / subset / item["file"]
        )

    chosen_speakers = list(spk_files.keys())[:max_speakers]
    raw_embs, spk_embs, spk_labels = [], [], []

    for i, spk in enumerate(chosen_speakers):
        files = spk_files[spk][:max_per_speaker]
        for f in files:
            emb = torch.load(f, weights_only=True).float()
            raw_embs.append(emb.numpy())
            if model is not None:
                with torch.no_grad():
                    e_spk, _ = model.encode(emb.unsqueeze(0))
                spk_embs.append(e_spk.squeeze(0).numpy())
            spk_labels.append(i)

    return np.array(raw_embs), np.array(spk_embs) if spk_embs else None, np.array(spk_labels)


def plot_tsne(raw_embs, spk_embs, labels, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    raw_2d = reducer.fit_transform(raw_embs)
    axes[0].scatter(raw_2d[:, 0], raw_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=15)
    axes[0].set_title("Raw Embeddings (Baseline)")
    axes[0].axis("off")

    if spk_embs is not None:
        reducer2 = TSNE(n_components=2, random_state=42, perplexity=30)
        spk_2d = reducer2.fit_transform(spk_embs)
        axes[1].scatter(spk_2d[:, 0], spk_2d[:, 1], c=labels, cmap="tab20", alpha=0.7, s=15)
        axes[1].set_title("Disentangled Speaker Embeddings (Proposed)")
        axes[1].axis("off")
    else:
        axes[1].set_visible(False)

    plt.suptitle("t-SNE: Speaker Separability", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_improved", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    emb_dim = config["model"]["embedding_dim"]
    latent_dim = config["model"]["latent_dim"]
    env_hidden = config["model"]["env_disc_hidden"]
    embedding_dir = config["data"]["embedding_dir"]
    test_subset = config["data"]["test_subset"]
    results_dir = Path(config["eval"]["results_dir"])
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    num_trials = config["eval"]["num_trials"]
    trials = build_verification_trials(embedding_dir, test_subset, num_trials)
    print(f"Generated {len(trials)} verification trials")

    ckpt_path = args.checkpoint or config["eval"].get("checkpoint")
    ckpt_improved_path = args.checkpoint_improved or config["eval"].get("checkpoint_improved")

    model_proposed = None
    model_improved = None
    num_speakers = 1

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        num_speakers = len(ckpt["speaker_to_idx"])
        model_proposed = DisentanglementModel(emb_dim, latent_dim, env_hidden, num_speakers)
        model_proposed.load_state_dict(ckpt["model_state"])
        model_proposed.eval()
        print(f"Loaded proposed model from {ckpt_path}")

    if ckpt_improved_path and Path(ckpt_improved_path).exists():
        ckpt_imp = torch.load(ckpt_improved_path, map_location="cpu", weights_only=False)
        if num_speakers == 1:
            num_speakers = len(ckpt_imp["speaker_to_idx"])
        model_improved = DisentanglementModel(emb_dim, latent_dim, env_hidden, num_speakers)
        model_improved.load_state_dict(ckpt_imp["model_state"])
        model_improved.eval()
        print(f"Loaded improved model from {ckpt_improved_path}")

    print("Scoring: Baseline...")
    base_labels, base_scores = score_trials(trials, model=None)

    all_results = {"Baseline": (base_labels, base_scores)}
    metrics = {}

    eer_base, _ = compute_eer(base_labels, base_scores)
    dcf_base = compute_min_dcf(base_labels, base_scores)
    metrics["Baseline"] = {"EER (%)": round(eer_base, 3), "minDCF": round(dcf_base, 4)}
    print(f"Baseline   EER={eer_base:.3f}%  minDCF={dcf_base:.4f}")

    if model_proposed is not None:
        print("Scoring: Proposed...")
        prop_labels, prop_scores = score_trials(trials, model=model_proposed)
        all_results["Proposed"] = (prop_labels, prop_scores)
        eer_prop, _ = compute_eer(prop_labels, prop_scores)
        dcf_prop = compute_min_dcf(prop_labels, prop_scores)
        metrics["Proposed"] = {"EER (%)": round(eer_prop, 3), "minDCF": round(dcf_prop, 4)}
        print(f"Proposed   EER={eer_prop:.3f}%  minDCF={dcf_prop:.4f}")

    if model_improved is not None:
        print("Scoring: Improved...")
        imp_labels, imp_scores = score_trials(trials, model=model_improved)
        all_results["Improved"] = (imp_labels, imp_scores)
        eer_imp, _ = compute_eer(imp_labels, imp_scores)
        dcf_imp = compute_min_dcf(imp_labels, imp_scores)
        metrics["Improved"] = {"EER (%)": round(eer_imp, 3), "minDCF": round(dcf_imp, 4)}
        print(f"Improved   EER={eer_imp:.3f}%  minDCF={dcf_imp:.4f}")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Results Table ---")
    print(f"{'System':<15} {'EER (%)':<12} {'minDCF':<10}")
    print("-" * 37)
    for sys_name, m in metrics.items():
        print(f"{sys_name:<15} {m['EER (%)']:<12.3f} {m['minDCF']:<10.4f}")

    print("\nGenerating plots...")
    plot_det_curve(all_results, plots_dir / "det_curve.png")
    plot_score_distributions(all_results, plots_dir / "score_distributions.png")

    raw_embs, spk_embs, labels = collect_embeddings_for_tsne(
        embedding_dir, test_subset, model_proposed, max_speakers=20
    )
    plot_tsne(raw_embs, spk_embs, labels, plots_dir / "tsne.png")

    history_paths = []
    hist_proposed = results_dir / "train_history_proposed.json"
    hist_improved = results_dir / "train_history_improved.json"
    if hist_proposed.exists():
        history_paths.append(("Proposed", hist_proposed))
    if hist_improved.exists():
        history_paths.append(("Improved", hist_improved))
    if history_paths:
        plot_training_curves(history_paths, plots_dir / "training_curves.png")

    print(f"\nResults saved to {results_dir}")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
