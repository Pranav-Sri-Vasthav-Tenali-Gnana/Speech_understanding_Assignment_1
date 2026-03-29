# Q2: Disentangled Representation Learning for Environment-agnostic Speaker Recognition

Implementation based on: https://arxiv.org/abs/2406.14559

---

## Overview

The framework trains a lightweight disentangled auto-encoder on top of pre-extracted speaker embeddings. The encoder splits each embedding into a speaker component `e_spk` and an environment component `e_env`. Only `e_spk` is used at verification time. Three losses drive the disentanglement: a speaker classification loss, an environment triplet loss, and an adversarial loss that prevents environment information from leaking into `e_spk`. A correlation penalty (or batch cross-correlation in the improved variant) provides additional disentanglement regularisation.

---

## Directory Structure

```
q2/
├── configs/
│   ├── config.yaml          # default config (MFCC extractor, fast)
│   └── ecapa_config.yaml    # ECAPA-TDNN extractor (accurate, slow)
├── results/
│   ├── checkpoints/         # saved model checkpoints
│   ├── plots/               # DET curve, score distributions, t-SNE, loss curves
│   ├── metrics.json         # EER and minDCF table
│   ├── train_history_proposed.json
│   └── train_history_improved.json
├── models.py                # Encoder, Decoder, EnvironmentMLP, DisentanglementModel
├── dataset.py               # TripletDataset, VerificationDataset, trial generation
├── train.py                 # embedding extraction + training
├── eval.py                  # scoring, EER, minDCF, plots
├── review.md                # technical critical review (convert to PDF)
└── q2_readme.md
```

---

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
numpy
scipy
scikit-learn
matplotlib
pyyaml
```

For ECAPA extractor (optional, requires internet + ~300 MB download):
```
speechbrain
```

Install all dependencies:
```bash
pip install torch torchaudio numpy scipy scikit-learn matplotlib pyyaml
pip install speechbrain   # optional
```

---

## Dataset

LibriSpeech train-clean-100 and train-clean-360 are used for training.
LibriSpeech test-clean is used for evaluation.

Set `data.root` in `configs/config.yaml` to the directory where LibriSpeech should be downloaded.
The datasets are downloaded automatically via `torchaudio` if not already present.

Expected structure after download:
```
data/LibriSpeech/
├── train-clean-100/
├── train-clean-360/
└── test-clean/
```

---

## Step 1: Extract Embeddings

**Default (MFCC statistics, fast, ~10 minutes for both splits):**
```bash
python train.py --config configs/config.yaml --phase extract
```

**ECAPA-TDNN (accurate, slow on CPU, ~8-12 hours for full dataset):**
```bash
python train.py --config configs/ecapa_config.yaml --phase extract
```

Embeddings are saved to `data/embeddings/` (or `data/embeddings_ecapa/` for ECAPA).
Extraction is resumable: already-extracted files are skipped on rerun.

**MFCC embedding details:**
- n_mfcc = 32, mean + std + delta_mean + delta_std + delta2_mean + delta2_std = 192-dim
- Fast and reproducible, no pre-trained model required

---

## Step 2: Train the Disentanglement Model

**Proposed method (Pearson correlation loss):**
```bash
python train.py --config configs/config.yaml --phase train
```

**Improved method (batch cross-correlation loss):**
```bash
python train.py --config configs/config.yaml --phase train --improved
```

Checkpoints are saved to `results/checkpoints/` after every epoch.
The best checkpoint (lowest total loss) is saved as `best_model.pt` or `best_model_improved.pt`.
Training history is saved as `results/train_history_proposed.json` or `train_history_improved.json`.

Training typically converges in 50–100 epochs.

---

## Step 3: Evaluate

```bash
python eval.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --checkpoint_improved results/checkpoints/best_model_improved.pt
```

This produces:
- `results/metrics.json` — EER (%) and minDCF for Baseline, Proposed, and Improved
- `results/plots/det_curve.png` — DET curve comparison
- `results/plots/score_distributions.png` — positive vs negative score histograms
- `results/plots/tsne.png` — t-SNE of raw vs disentangled speaker embeddings
- `results/plots/training_curves.png` — loss curves per epoch

---

## Checkpoint Correspondence

| File | Description |
|------|-------------|
| `results/checkpoints/best_model.pt` | Best proposed model (lowest total loss, Pearson correlation) |
| `results/checkpoints/best_model_improved.pt` | Best improved model (batch cross-correlation loss) |
| `results/checkpoints/checkpoint_epoch*.pt` | Per-epoch checkpoints for proposed |
| `results/checkpoints/checkpoint_epoch*_improved.pt` | Per-epoch checkpoints for improved |

The `metrics.json` table reports results from `best_model.pt` and `best_model_improved.pt`.

---

## Reduced Reproduction Notes

The original paper trains on VoxCeleb2 (5994 speakers, video session metadata for environment-aware triplets). This implementation uses LibriSpeech as instructed, with the following adaptations:

- **Environment proxy**: Book chapters (`chapter_id`) substitute for video sessions. Different chapters from the same speaker form the positive environment pair; different chapters serve as the negative.
- **Embedding extractor**: MFCC statistics (192-dim) by default. Switch to ECAPA-TDNN via `ecapa_config.yaml` for closer reproduction of the paper.
- **Backbone training**: The paper jointly trains the embedding extractor and the AE. Here only the AE is trained, with the backbone frozen. This reduces compute significantly and is the primary difference from the paper's setup.

---

## Improvement

The improved model replaces the sample-wise Pearson correlation loss with a batch-level cross-correlation loss (see `review.md` Section 7). This enforces that every feature dimension of `e_spk` is uncorrelated with every feature dimension of `e_env`, providing stronger disentanglement than scalar Pearson correlation. Enable it with `--improved` in training.
