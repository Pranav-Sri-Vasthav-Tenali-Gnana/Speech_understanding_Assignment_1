import json
import random
import torch
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, embedding_dir, subsets, speaker_to_idx, max_speakers=None):
        self.embedding_dir = Path(embedding_dir)
        self.speaker_to_idx = speaker_to_idx

        spk_chap_utts = defaultdict(lambda: defaultdict(list))

        for subset in subsets:
            meta_path = self.embedding_dir / f"{subset}_metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)
            for item in metadata:
                sid = item["speaker_id"]
                cid = item["chapter_id"]
                spk_chap_utts[sid][cid].append(
                    self.embedding_dir / subset / item["file"]
                )

        if max_speakers:
            keys = list(spk_chap_utts.keys())[:max_speakers]
            spk_chap_utts = {k: spk_chap_utts[k] for k in keys}

        self.triplets = []
        for sid, chap_dict in spk_chap_utts.items():
            valid_chaps = [c for c, utts in chap_dict.items() if len(utts) >= 2]
            if len(valid_chaps) < 2:
                continue
            for chap in valid_chaps:
                other_chaps = [c for c in valid_chaps if c != chap]
                if not other_chaps:
                    continue
                utts = chap_dict[chap]
                self.triplets.append({
                    "speaker_id": sid,
                    "chap_same": chap,
                    "utts_same": utts,
                    "chaps_diff": other_chaps,
                    "chap_dict": chap_dict,
                })

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        same_utts = item["utts_same"]
        u1, u2 = random.sample(same_utts, 2)

        diff_chap = random.choice(item["chaps_diff"])
        diff_utts = item["chap_dict"][diff_chap]
        u3 = random.choice(diff_utts)

        emb1 = torch.load(u1, weights_only=True).float()
        emb2 = torch.load(u2, weights_only=True).float()
        emb3 = torch.load(u3, weights_only=True).float()

        spk_idx = self.speaker_to_idx[item["speaker_id"]]
        return emb1, emb2, emb3, spk_idx


class VerificationDataset(Dataset):
    def __init__(self, embedding_dir, subset, trials):
        self.embedding_dir = Path(embedding_dir) / subset
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        path1, path2, label = self.trials[idx]
        emb1 = torch.load(path1, weights_only=True).float()
        emb2 = torch.load(path2, weights_only=True).float()
        return emb1, emb2, label


def build_speaker_to_idx(embedding_dir, subsets):
    speakers = set()
    for subset in subsets:
        meta_path = Path(embedding_dir) / f"{subset}_metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        for item in metadata:
            speakers.add(item["speaker_id"])
    return {sid: i for i, sid in enumerate(sorted(speakers))}


def build_verification_trials(embedding_dir, subset, num_trials, seed=42):
    random.seed(seed)
    meta_path = Path(embedding_dir) / f"{subset}_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    spk_to_files = defaultdict(list)
    for item in metadata:
        path = Path(embedding_dir) / subset / item["file"]
        spk_to_files[item["speaker_id"]].append(path)

    speakers = [s for s, files in spk_to_files.items() if len(files) >= 2]
    trials = []

    half = num_trials // 2

    while len([t for t in trials if t[2] == 1]) < half:
        spk = random.choice(speakers)
        files = spk_to_files[spk]
        if len(files) < 2:
            continue
        f1, f2 = random.sample(files, 2)
        trials.append((f1, f2, 1))

    while len([t for t in trials if t[2] == 0]) < half:
        s1, s2 = random.sample(speakers, 2)
        f1 = random.choice(spk_to_files[s1])
        f2 = random.choice(spk_to_files[s2])
        trials.append((f1, f2, 0))

    random.shuffle(trials)
    return trials
