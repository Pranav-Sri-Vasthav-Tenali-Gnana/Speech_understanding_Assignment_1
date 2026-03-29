# Q1 — Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

## Directory Structure

```
q1/
├── mfcc_manual.py       # Step 1: Handcrafted MFCC / Cepstrum engine
├── leakage_snr.py       # Step 2: Spectral leakage & SNR analysis
├── voiced_unvoiced.py   # Step 3: Voiced/unvoiced boundary detection
├── phonetic_mapping.py  # Step 4: Wav2Vec2 forced alignment + RMSE
├── requirements.txt
├── data/
│   └── manifest.txt     # Audio file sources
└── outputs/             # Created automatically by the scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### 1. MFCC Extraction
```bash
python mfcc_manual.py \
    --audio data/cv-sample-01.wav \
    --n_mfcc 13 \
    --n_filters 26 \
    --n_fft 512 \
    --window hamming \
    --save_dir outputs/
```
Outputs: `mfcc_plot.png`, `filterbank_plot.png`, `mfcc.npy`

### 2. Spectral Leakage & SNR
```bash
python leakage_snr.py \
    --audio data/cv-sample-01.wav \
    --segment_start 0.5 \
    --segment_end 1.5 \
    --n_fft 2048 \
    --save_dir outputs/
```
Outputs: `leakage_snr_table.csv`, `window_shapes.png`, `spectra_comparison.png`,
         `snr_comparison.png`, `leakage_comparison.png`

### 3. Voiced / Unvoiced Detection
```bash
python voiced_unvoiced.py \
    --audio data/cv-sample-01.wav \
    --frame_ms 25 \
    --hop_ms 10 \
    --pitch_min 60 \
    --pitch_max 400 \
    --threshold 0.45 \
    --save_dir outputs/
```
Outputs: `voiced_unvoiced.png`, `segments.csv`, `frame_labels.npy`

### 4. Phonetic Mapping & RMSE
```bash
python phonetic_mapping.py \
    --audio   data/cv-sample-01.wav \
    --segments outputs/segments.csv \
    --save_dir outputs/
```
Outputs: `phone_mapping.csv`, `rmse_summary.json`, `phone_alignment.png`

## Key Hyperparameters

| Parameter         | Default | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `n_mfcc`          | 13      | Number of MFCC coefficients to retain            |
| `n_filters`       | 26      | Number of Mel filterbank channels                |
| `n_fft`           | 512     | FFT size                                         |
| `frame_ms`        | 25 ms   | Analysis frame duration                          |
| `hop_ms`          | 10 ms   | Frame hop (controls overlap)                     |
| `pre_emph_coef`   | 0.97    | Pre-emphasis filter coefficient                  |
| `pitch_min_hz`    | 60 Hz   | Lower bound of voiced pitch range                |
| `pitch_max_hz`    | 400 Hz  | Upper bound of voiced pitch range                |
| `threshold`       | 0.45    | Voiced probability threshold for classification  |
| Wav2Vec2 model    | facebook/wav2vec2-base-960h | HuggingFace model for forced alignment |

## Notes

- All scripts write their outputs to `--save_dir` (default `.`).
- `phonetic_mapping.py` requires `segments.csv` produced by `voiced_unvoiced.py`.
- The first run of `phonetic_mapping.py` will download the Wav2Vec2 model (~360 MB).
- For best SNR estimation, choose a segment containing both speech and quiet noise.
