import os
import sys
import json
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

from models import DisentanglementModel
from dataset import TripletDataset, build_speaker_to_idx


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_mfcc_embedding(waveform, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=32,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
    )
    mfcc = mfcc_transform(waveform).squeeze(0)

    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    feat = torch.cat([
        mfcc.mean(dim=1),
        mfcc.std(dim=1),
        delta.mean(dim=1),
        delta.std(dim=1),
        delta2.mean(dim=1),
        delta2.std(dim=1),
    ])
    return feat


def extract_ecapa_embedding(waveform, sample_rate, classifier):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform.unsqueeze(0))
    return embedding.squeeze()


def load_flac(flac_path):
    data, sr = sf.read(str(flac_path), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return waveform, sr


def scan_librispeech_subset(root, subset):
    root = Path(root)
    candidate = root / subset
    if not candidate.exists():
        candidate = root / "LibriSpeech" / subset
    entries = []
    for flac_path in sorted(candidate.rglob("*.flac")):
        parts = flac_path.stem.split("-")
        if len(parts) != 3:
            continue
        speaker_id, chapter_id, utt_id = int(parts[0]), int(parts[1]), int(parts[2])
        entries.append((flac_path, speaker_id, chapter_id, utt_id))
    return entries


def extract_embeddings(config):
    embedding_dir = Path(config["data"]["embedding_dir"])
    embedding_dir.mkdir(parents=True, exist_ok=True)

    extractor_type = config["data"].get("extractor", "mfcc")

    classifier = None
    if extractor_type == "ecapa":
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )

    librispeech_root = Path(config["data"]["root"])
    librispeech_root.mkdir(parents=True, exist_ok=True)

    all_subsets = config["data"]["subsets"] + [config["data"]["test_subset"]]

    for subset in all_subsets:
        subset_dir = embedding_dir / subset
        subset_dir.mkdir(exist_ok=True)

        meta_path = embedding_dir / f"{subset}_metadata.json"
        if meta_path.exists():
            print(f"Skipping {subset}, metadata already exists.")
            continue

        direct_path = librispeech_root / subset
        nested_path = librispeech_root / "LibriSpeech" / subset
        if not direct_path.exists() and not nested_path.exists():
            print(f"Downloading {subset}...")
            torchaudio.datasets.LIBRISPEECH(
                root=str(librispeech_root),
                url=subset,
                download=True,
            )

        print(f"Extracting {subset}...")
        entries = scan_librispeech_subset(librispeech_root, subset)
        print(f"  Found {len(entries)} FLAC files")

        metadata = []
        for idx, (flac_path, speaker_id, chapter_id, utt_id) in enumerate(entries):
            filename = f"{speaker_id}_{chapter_id}_{utt_id}.pt"
            out_path = subset_dir / filename

            if not out_path.exists():
                waveform, sr = load_flac(flac_path)
                if extractor_type == "ecapa":
                    emb = extract_ecapa_embedding(waveform, sr, classifier)
                else:
                    emb = extract_mfcc_embedding(waveform, sr)
                torch.save(emb.cpu(), out_path)

            metadata.append({
                "file": filename,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utt_id": utt_id,
            })

            if idx % 500 == 0:
                print(f"  {subset}: {idx}/{len(entries)}")

        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        print(f"  Done: {len(metadata)} utterances")


def triplet_loss(anchor, positive, negative, margin):
    pos_dist = (anchor - positive).pow(2).sum(dim=1)
    neg_dist = (anchor - negative).pow(2).sum(dim=1)
    loss = F.relu(margin + pos_dist - neg_dist).mean()
    return loss, pos_dist.mean().item(), neg_dist.mean().item()


def correlation_loss(e_spk, e_env):
    spk_c = e_spk - e_spk.mean(dim=0)
    env_c = e_env - e_env.mean(dim=0)
    spk_std = spk_c.std(dim=0).clamp(min=1e-8)
    env_std = env_c.std(dim=0).clamp(min=1e-8)
    corr = (spk_c / spk_std * env_c / env_std).mean(dim=0)
    return corr.abs().mean()


def batch_cross_correlation_loss(e_spk, e_env):
    N = e_spk.size(0)
    D = e_spk.size(1)
    spk_norm = (e_spk - e_spk.mean(0)) / (e_spk.std(0) + 1e-8)
    env_norm = (e_env - e_env.mean(0)) / (e_env.std(0) + 1e-8)
    C = (spk_norm.T @ env_norm) / N
    return (C ** 2).sum() / (D * D)


def train_epoch(model, loader, main_optimizer, adv_optimizer, config, improved=False):
    model.train()
    cfg = config["training"]
    margin = cfg["margin"]
    lam_spk = cfg["lambda_spk"]
    lam_r = cfg["lambda_recons"]
    lam_env = cfg["lambda_env"]
    lam_adv = cfg["lambda_adv"]
    lam_c = cfg["lambda_corr"]

    total_loss = 0.0
    total_spk = 0.0
    total_recons = 0.0
    total_env = 0.0
    total_adv = 0.0
    total_corr = 0.0
    n_batches = 0

    for emb1, emb2, emb3, spk_ids in loader:
        e_spk1_d, _ = model.encode(emb1)
        e_spk2_d, _ = model.encode(emb2)
        e_spk3_d, _ = model.encode(emb3)

        adv1 = model.adv_disc(e_spk1_d.detach())
        adv2 = model.adv_disc(e_spk2_d.detach())
        adv3 = model.adv_disc(e_spk3_d.detach())

        L_disc, _, _ = triplet_loss(adv1, adv2, adv3, margin)
        adv_optimizer.zero_grad()
        L_disc.backward()
        adv_optimizer.step()

        e_spk1, e_env1 = model.encode(emb1)
        e_spk2, e_env2 = model.encode(emb2)
        e_spk3, e_env3 = model.encode(emb3)

        recon1 = model.decode(e_spk1, e_env1)
        recon2 = model.decode(e_spk2, e_env2)
        recon3 = model.decode(e_spk3, e_env3)

        L_R = (F.l1_loss(recon1, emb1) + F.l1_loss(recon2, emb2) + F.l1_loss(recon3, emb3)) / 3

        all_e_spk = torch.cat([e_spk1, e_spk2, e_spk3])
        all_labels = torch.cat([spk_ids, spk_ids, spk_ids])
        spk_logits = model.speaker_disc(all_e_spk)
        L_spk = F.cross_entropy(spk_logits, all_labels)

        env1_proj = model.env_disc(e_env1)
        env2_proj = model.env_disc(e_env2)
        env3_proj = model.env_disc(e_env3)
        L_env, _, _ = triplet_loss(env1_proj, env2_proj, env3_proj, margin)

        if improved:
            all_e_spk_cat = torch.cat([e_spk1, e_spk2, e_spk3])
            all_e_env_cat = torch.cat([e_env1, e_env2, e_env3])
            L_C = batch_cross_correlation_loss(all_e_spk_cat, all_e_env_cat)
        else:
            L_C = (
                correlation_loss(e_spk1, e_env1)
                + correlation_loss(e_spk2, e_env2)
                + correlation_loss(e_spk3, e_env3)
            ) / 3

        adv1_g = model.adv_disc(e_spk1)
        adv2_g = model.adv_disc(e_spk2)
        adv3_g = model.adv_disc(e_spk3)

        pos_dist_adv = (adv1_g - adv2_g).pow(2).sum(dim=1)
        neg_dist_adv = (adv1_g - adv3_g).pow(2).sum(dim=1)
        L_adv = (pos_dist_adv - neg_dist_adv).mean()

        L_total = (
            lam_spk * L_spk
            + lam_r * L_R
            + lam_env * L_env
            + lam_adv * L_adv
            + lam_c * L_C
        )

        main_optimizer.zero_grad()
        L_total.backward()
        main_optimizer.step()

        total_loss += L_total.item()
        total_spk += L_spk.item()
        total_recons += L_R.item()
        total_env += L_env.item()
        total_adv += L_adv.item()
        total_corr += L_C.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "total": total_loss / n,
        "spk": total_spk / n,
        "recons": total_recons / n,
        "env": total_env / n,
        "adv": total_adv / n,
        "corr": total_corr / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phase", choices=["extract", "train", "all"], default="all")
    parser.add_argument("--improved", action="store_true", help="Use batch cross-correlation loss instead of Pearson")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    if args.phase in ("extract", "all"):
        extract_embeddings(config)

    if args.phase in ("train", "all"):
        embedding_dir = config["data"]["embedding_dir"]
        subsets = config["data"]["subsets"]

        speaker_to_idx = build_speaker_to_idx(embedding_dir, subsets)
        num_speakers = len(speaker_to_idx)
        print(f"Total speakers: {num_speakers}")

        max_spk = config["data"].get("max_speakers")
        dataset = TripletDataset(embedding_dir, subsets, speaker_to_idx, max_speakers=max_spk)
        print(f"Total triplets: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        emb_dim = config["model"]["embedding_dim"]
        latent_dim = config["model"]["latent_dim"]
        env_hidden = config["model"]["env_disc_hidden"]

        model = DisentanglementModel(emb_dim, latent_dim, env_hidden, num_speakers)

        main_params = (
            list(model.encoder.parameters())
            + list(model.decoder.parameters())
            + list(model.speaker_disc.parameters())
            + list(model.env_disc.parameters())
        )
        adv_params = list(model.adv_disc.parameters())

        lr = config["training"]["lr"]
        main_optimizer = torch.optim.Adam(main_params, lr=lr)
        adv_optimizer = torch.optim.Adam(adv_params, lr=lr)

        decay = config["training"]["lr_decay_factor"]
        step = config["training"]["lr_decay_epochs"]
        main_scheduler = torch.optim.lr_scheduler.StepLR(main_optimizer, step_size=step, gamma=decay)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=step, gamma=decay)

        ckpt_dir = Path(config["training"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history = []
        best_loss = float("inf")

        for epoch in range(1, config["training"]["epochs"] + 1):
            losses = train_epoch(model, loader, main_optimizer, adv_optimizer, config, improved=args.improved)
            main_scheduler.step()
            adv_scheduler.step()

            history.append({"epoch": epoch, **losses})

            print(
                f"Epoch {epoch:3d} | total={losses['total']:.4f} spk={losses['spk']:.4f} "
                f"recons={losses['recons']:.4f} env={losses['env']:.4f} "
                f"adv={losses['adv']:.4f} corr={losses['corr']:.4f}"
            )

            suffix = "_improved" if args.improved else ""
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "speaker_to_idx": speaker_to_idx,
                    "config": config,
                },
                ckpt_dir / f"checkpoint_epoch{epoch}{suffix}.pt",
            )

            if losses["total"] < best_loss:
                best_loss = losses["total"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "speaker_to_idx": speaker_to_idx,
                        "config": config,
                    },
                    ckpt_dir / f"best_model{suffix}.pt",
                )

        tag = "improved" if args.improved else "proposed"
        with open(Path(config["training"]["checkpoint_dir"]).parent / f"train_history_{tag}.json", "w") as f:
            json.dump(history, f, indent=2)

        print("Training complete.")


if __name__ == "__main__":
    main()
