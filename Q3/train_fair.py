import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


SPEAKERS_TXT = "../Q2/data/LibriSpeech/LibriSpeech/SPEAKERS.TXT"
EMBEDDING_DIR = "../Q2/data/embeddings"
SUBSETS = ["train-clean-100", "train-clean-360"]
TEST_SUBSET = "test-clean"


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
            g = parts[1].upper()
            if g in ("M", "F"):
                gender_map[sid] = g
    return gender_map


class GenderDataset(Dataset):
    def __init__(self, embedding_dir, subsets, gender_map, max_per_subset=None):
        self.samples = []
        self.embeddings = []
        self.labels = []
        emb_dir = Path(embedding_dir)
        for subset in subsets:
            meta_path = emb_dir / f"{subset}_metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                metadata = json.load(f)
            count = 0
            for item in metadata:
                sid = item["speaker_id"]
                if sid not in gender_map:
                    continue
                emb_path = emb_dir / subset / item["file"]
                if not emb_path.exists():
                    continue
                label = 0 if gender_map[sid] == "M" else 1
                self.samples.append((emb_path, label, sid))
                count += 1
                if max_per_subset and count >= max_per_subset:
                    break
        print(f"  Loading {len(self.samples)} embeddings into RAM...")
        for path, label, sid in self.samples:
            self.embeddings.append(torch.load(path, weights_only=True).float())
            self.labels.append(label)
        self.embeddings = torch.stack(self.embeddings)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"  Done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class GenderClassifier(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def fairness_loss(logits, labels, lambda_fair):
    male_mask = labels == 0
    female_mask = labels == 1

    if male_mask.sum() == 0 or female_mask.sum() == 0:
        return torch.tensor(0.0)

    loss_male = F.cross_entropy(logits[male_mask], labels[male_mask])
    loss_female = F.cross_entropy(logits[female_mask], labels[female_mask])

    equalization_penalty = (loss_male - loss_female).abs()
    return lambda_fair * equalization_penalty


def accuracy_by_group(logits, labels):
    preds = logits.argmax(dim=1)
    male_mask = labels == 0
    female_mask = labels == 1

    acc_m = (preds[male_mask] == labels[male_mask]).float().mean().item() if male_mask.sum() > 0 else 0.0
    acc_f = (preds[female_mask] == labels[female_mask]).float().mean().item() if female_mask.sum() > 0 else 0.0
    acc_total = (preds == labels).float().mean().item()
    return acc_total, acc_m, acc_f


def train_one_epoch(model, loader, optimizer, use_fairness, lambda_fair):
    model.train()
    total_loss = 0.0
    total_fair = 0.0
    n = 0

    for embs, labels in loader:
        logits = model(embs)
        ce_loss = F.cross_entropy(logits, labels)
        f_loss = fairness_loss(logits, labels, lambda_fair) if use_fairness else torch.tensor(0.0)
        loss = ce_loss + f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += ce_loss.item()
        total_fair += f_loss.item()
        n += 1

    return total_loss / max(n, 1), total_fair / max(n, 1)


def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for embs, labels in loader:
            logits = model(embs)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, acc_m, acc_f = accuracy_by_group(all_logits, all_labels)
    gap = abs(acc_m - acc_f)
    return acc, acc_m, acc_f, gap


def plot_training_curves(history_std, history_fair, out_dir):
    out_dir = Path(out_dir)
    epochs = [h["epoch"] for h in history_std]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for hist, label, color in [(history_std, "Standard", "#4C72B0"), (history_fair, "Fair", "#DD8452")]:
        epochs_h = [h["epoch"] for h in hist]
        axes[0].plot(epochs_h, [h["train_loss"] for h in hist], label=label, color=color)
        axes[1].plot(epochs_h, [h["test_acc"] for h in hist], label=label, color=color)
        axes[2].plot(epochs_h, [h["gap"] for h in hist], label=label, color=color)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].set_title("Gender Accuracy Gap |Acc_M - Acc_F|")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Standard vs Fair Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fairness_training_curves.png", dpi=150)
    plt.close()


def plot_group_accuracy_bars(std_result, fair_result, out_dir):
    out_dir = Path(out_dir)
    labels = ["Overall", "Male", "Female"]
    std_vals = [std_result["test_acc"], std_result["acc_male"], std_result["acc_female"]]
    fair_vals = [fair_result["test_acc"], fair_result["acc_male"], fair_result["acc_female"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, std_vals, width, label="Standard Training", color="#4C72B0")
    ax.bar(x + width / 2, fair_vals, width, label="Fair Training", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Per-Group Accuracy: Standard vs Fair Training")
    ax.grid(axis="y", alpha=0.3)
    for i, (sv, fv) in enumerate(zip(std_vals, fair_vals)):
        ax.text(i - width / 2, sv + 0.01, f"{sv:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, fv + 0.01, f"{fv:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "group_accuracy_comparison.png", dpi=150)
    plt.close()


def run_training(train_dataset, test_dataset, use_fairness, lambda_fair, epochs, batch_size, lr, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = GenderClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, fair_penalty = train_one_epoch(model, train_loader, optimizer, use_fairness, lambda_fair)
        scheduler.step()
        acc, acc_m, acc_f, gap = evaluate(model, test_loader)

        tag = "fair" if use_fairness else "std"
        print(
            f"[{tag}] Epoch {epoch:3d} | loss={train_loss:.4f} fair={fair_penalty:.4f} "
            f"acc={acc:.4f} acc_M={acc_m:.4f} acc_F={acc_f:.4f} gap={gap:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "fair_penalty": fair_penalty,
            "test_acc": acc,
            "acc_male": acc_m,
            "acc_female": acc_f,
            "gap": gap,
        })

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers_txt", default=SPEAKERS_TXT)
    parser.add_argument("--embedding_dir", default=EMBEDDING_DIR)
    parser.add_argument("--train_subsets", nargs="+", default=SUBSETS)
    parser.add_argument("--test_subset", default=TEST_SUBSET)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_fair", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--max_per_subset", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gender_map = parse_gender_map(args.speakers_txt)
    print(f"Loaded gender labels for {len(gender_map)} speakers")

    train_dataset = GenderDataset(args.embedding_dir, args.train_subsets, gender_map, args.max_per_subset)
    test_dataset = GenderDataset(args.embedding_dir, [args.test_subset], gender_map, args.max_per_subset)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    male_train = sum(1 for _, l, _ in train_dataset.samples if l == 0)
    female_train = sum(1 for _, l, _ in train_dataset.samples if l == 1)
    print(f"Train — Male: {male_train}, Female: {female_train}")

    print("\n--- Standard Training ---")
    model_std, history_std = run_training(
        train_dataset, test_dataset,
        use_fairness=False, lambda_fair=0.0,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )

    print("\n--- Fair Training ---")
    model_fair, history_fair = run_training(
        train_dataset, test_dataset,
        use_fairness=True, lambda_fair=args.lambda_fair,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )

    std_final = history_std[-1]
    fair_final = history_fair[-1]

    print("\n=== Final Results ===")
    print(f"{'System':<20} {'Acc':>6} {'Acc_M':>7} {'Acc_F':>7} {'Gap':>7}")
    print("-" * 50)
    print(f"{'Standard':<20} {std_final['test_acc']:>6.4f} {std_final['acc_male']:>7.4f} "
          f"{std_final['acc_female']:>7.4f} {std_final['gap']:>7.4f}")
    print(f"{'Fair (λ={:.1f})'.format(args.lambda_fair):<20} {fair_final['test_acc']:>6.4f} "
          f"{fair_final['acc_male']:>7.4f} {fair_final['acc_female']:>7.4f} {fair_final['gap']:>7.4f}")

    std_result = {"test_acc": std_final["test_acc"], "acc_male": std_final["acc_male"],
                  "acc_female": std_final["acc_female"], "gap": std_final["gap"]}
    fair_result = {"test_acc": fair_final["test_acc"], "acc_male": fair_final["acc_male"],
                   "acc_female": fair_final["acc_female"], "gap": fair_final["gap"]}

    with open(out_dir / "fairness_results.json", "w") as f:
        json.dump({"standard": std_result, "fair": fair_result,
                   "lambda_fair": args.lambda_fair}, f, indent=2)

    with open(out_dir / "train_history_std.json", "w") as f:
        json.dump(history_std, f, indent=2)
    with open(out_dir / "train_history_fair.json", "w") as f:
        json.dump(history_fair, f, indent=2)

    torch.save(model_std.state_dict(), out_dir / "model_std.pt")
    torch.save(model_fair.state_dict(), out_dir / "model_fair.pt")

    plot_training_curves(history_std, history_fair, plots_dir)
    plot_group_accuracy_bars(std_result, fair_result, plots_dir)
    print(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    main()
