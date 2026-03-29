# Q3: Ethical Auditing & Documentation Debt Mitigation

## Overview

This module audits LibriSpeech for representation bias, implements a privacy-preserving voice transformation module, trains a gender classifier with a fairness loss, and validates the transformations using proxy quality metrics.

## Directory Structure

```
q3/
├── audit.py                       # Bias audit of LibriSpeech
├── privacymodule.py               # Privacy-preserving voice transformation
├── pp_demo.py                     # Demo: generate before/after audio pairs
├── train_fair.py                  # Fair vs standard gender classifier training
├── evaluation_scripts/
│   ├── fad_proxy.py               # Fréchet Audio Distance proxy
│   └── dnsmos_proxy.py            # DNSMOS-style quality proxy
├── examples/                      # Generated audio pairs (by pp_demo.py)
├── results/
│   ├── plots/                     # All generated plots
│   ├── audit_report.json
│   ├── fairness_results.json
│   ├── fad_results.json
│   └── dnsmos_results.json
└── q3_readme.md
```

## Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
numpy
scipy
matplotlib
soundfile
```

All are available from the Q2 setup. No additional installs needed.

## Step 1: Run the Bias Audit

```bash
python audit.py
```

Parses LibriSpeech SPEAKERS.TXT and produces:
- Gender distribution stats (speaker counts, speaking time)
- Per-subset breakdown
- Documentation debt report
- 3 plots: gender_distribution.png, subset_breakdown.png, imbalance_heatmap.png
- results/audit_report.json

## Step 2: Generate Privacy-Preserved Audio Pairs

```bash
python pp_demo.py
```

Picks 3 male and 3 female speakers from train-clean-100 and applies voice transformation:
- Male → Female: pitch factor 1.25x, formant shift 1.125x
- Female → Male: pitch factor 0.80x, formant shift 0.90x

Saves before/after WAV pairs to examples/ and waveform/spectrogram comparison plots.

## Step 3: Train Fair vs Standard Classifier

```bash
python train_fair.py
```

Trains a 3-layer MLP gender classifier twice:
1. Standard cross-entropy training
2. Same model + fairness loss: λ · |L_male − L_female|

Produces training curves and per-group accuracy comparison plots.

## Step 4: Evaluate Audio Quality

```bash
# FAD proxy — compares MFCC distributions of original vs transformed audio
python evaluation_scripts/fad_proxy.py --original_dir ../Q2/data/LibriSpeech/LibriSpeech/train-clean-100

# DNSMOS proxy — per-file quality scores for the generated examples
python evaluation_scripts/dnsmos_proxy.py
```

## Privacy Module Details

`PrivacyPreservingModule` (privacymodule.py) applies two steps:
1. **Pitch shift** via rate-conversion (resample to modified frequency, resample back)
2. **Formant shift** via STFT spectral envelope interpolation

Both steps are pure PyTorch / torchaudio — no FFmpeg, no external libraries.

Transformation parameters:
| Direction | Pitch Factor | Formant Factor |
|-----------|-------------|----------------|
| M → F     | 1.25x       | 1.125x         |
| F → M     | 0.80x       | 0.90x          |

RMS normalization is applied after transformation to preserve loudness.

## Fairness Loss

The fairness term in `train_fair.py` is an equalized-loss penalty:

```
L_fair = λ · |L_cross_entropy(male_samples) − L_cross_entropy(female_samples)|
```

This encourages the model to have equal cross-entropy loss (and thus similar accuracy) across gender groups, rather than trading off one group's performance for overall accuracy.

Default λ = 2.0. Adjust with `--lambda_fair`.

## Proxy Metrics

**FAD Proxy** (fad_proxy.py): Fréchet distance between 80-dim MFCC statistics of original vs transformed audio distributions. Lower = more similar distributions = less distortion.

**DNSMOS Proxy** (dnsmos_proxy.py): Estimates perceptual quality from signal-level features (SNR estimate, spectral flatness, zero-crossing rate, energy entropy) and combines them into a score on a 1–4 scale (matching DNSMOS conventions). Score ≥ 3.0 = acceptable quality.
