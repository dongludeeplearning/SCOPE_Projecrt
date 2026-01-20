# dataloader_relative.py
import os
import re
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ------------------------
# Transform (RGB 3-channel)
# ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class DAiSEEDataset(Dataset):
    """
    Pointwise dataset: each item is one clip folder (60 frames), with 4 task labels.
    Returns:
      - frames: (T, 3, 224, 224)
      - labels: (4,) long in {0,1,2,3}  order = [B,E,C,F]
      - optional au: (T, 41)
      - optional va: (T, 2)
    """
    def __init__(
        self,
        split,
        root_dir,
        label_file,
        au_root=None,
        va_root=None,
        include_au=False,
        include_va=False,
        transform_=None,
        expected_frames=60,
        img_size=224,
    ):
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.labels = pd.read_csv(label_file)
        self.labels.set_index("ClipID", inplace=True)

        self.transform = transform_ if transform_ is not None else transform
        self.include_au = include_au
        self.include_va = include_va
        self.au_root = au_root
        self.va_root = va_root
        self.expected_frames = expected_frames
        self.img_size = img_size

        if not os.path.exists(self.split_dir):
            print(f"[WARN] Directory not found: {self.split_dir}")
            self.video_folders = []
        else:
            self.video_folders = sorted(
                [f for f in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, f))]
            )

        # AU/VA split dirs
        self.au_split_dir = None
        self.va_split_dir = None
        if self.include_au and self.au_root:
            self.au_split_dir = os.path.join(self.au_root, split)
            if not os.path.exists(self.au_split_dir):
                print(f"[WARN] AU split dir not found: {self.au_split_dir}")

        if self.include_va and self.va_root:
            self.va_split_dir = os.path.join(self.va_root, split)
            if not os.path.exists(self.va_split_dir):
                print(f"[WARN] VA split dir not found: {self.va_split_dir}")

    def __len__(self):
        return len(self.video_folders)

    @staticmethod
    def _parse_au_string(au_string):
        # Parses string like "AU1: 47.38 ..." into floats
        values = re.findall(r":\s*([+-]?\d+(?:\.\d+)?)", au_string)
        if not values:
            values = re.findall(r"([+-]?\d+(?:\.\d+)?)", au_string)
        return [float(v) for v in values]

    def _get_au_features(self, video_id, frame_names):
        if not self.au_split_dir:
            return torch.zeros((len(frame_names), 41), dtype=torch.float32)

        json_path = os.path.join(self.au_split_dir, video_id, f"{video_id}.json")
        if not os.path.exists(json_path):
            return torch.zeros((len(frame_names), 41), dtype=torch.float32)

        feats = []
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for fn in frame_names:
                if fn in data and "Probabilities" in data[fn] and len(data[fn]["Probabilities"]) > 0:
                    raw = data[fn]["Probabilities"][0]
                    vec = self._parse_au_string(raw)
                    if len(vec) < 41:
                        vec = vec + [0.0] * (41 - len(vec))
                    elif len(vec) > 41:
                        vec = vec[:41]
                    feats.append(vec)
                else:
                    feats.append([0.0] * 41)

        except Exception:
            return torch.zeros((len(frame_names), 41), dtype=torch.float32)

        return torch.tensor(feats, dtype=torch.float32)

    def _get_va_features(self, video_id, frame_names):
        if not self.va_split_dir:
            return torch.zeros((len(frame_names), 2), dtype=torch.float32)

        video_dir = os.path.join(self.va_split_dir, video_id)
        if not os.path.exists(video_dir):
            return torch.zeros((len(frame_names), 2), dtype=torch.float32)

        feats = []
        for fn in frame_names:
            json_name = fn.replace(".jpg", ".json").replace(".png", ".json")
            jp = os.path.join(video_dir, json_name)
            val, aro = 0.0, 0.0
            if os.path.exists(jp):
                try:
                    with open(jp, "r") as f:
                        d = json.load(f)
                    if "Valence" in d:
                        val = float(d["Valence"])
                    elif "valence" in d:
                        val = float(d["valence"])
                    if "Arousal" in d:
                        aro = float(d["Arousal"])
                    elif "arousal" in d:
                        aro = float(d["arousal"])
                except Exception:
                    pass
            feats.append([val, aro])

        return torch.tensor(feats, dtype=torch.float32)

    def _load_frames(self, folder_path):
        if not os.path.exists(folder_path):
            return None, []

        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        if len(frame_files) == 0:
            return None, []

        frames = []
        for ff in frame_files:
            fp = os.path.join(folder_path, ff)
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img) if self.transform else torch.from_numpy(img)
            frames.append(img)

        if len(frames) == 0:
            return None, []

        frames = torch.stack(frames)  # (T,3,H,W)
        return frames, frame_files

    def _find_labels(self, video_id):
        # Normalize label index lookup (ClipID can be "xxx.avi" etc.)
        normalized_index = {str(idx).strip().lower(): idx for idx in self.labels.index}
        candidates = [str(video_id), f"{video_id}.avi", f"{video_id}.mp4", f"{video_id}.mkv"]

        resolved_key = None
        for cand in candidates:
            k = str(cand).strip().lower()
            if k in normalized_index:
                resolved_key = normalized_index[k]
                break
        if resolved_key is None:
            return None

        row = self.labels.loc[resolved_key, ["Boredom", "Engagement", "Confusion", "Frustration"]]
        return {
            "B": int(row["Boredom"]),
            "E": int(row["Engagement"]),
            "C": int(row["Confusion"]),
            "F": int(row["Frustration"]),
        }

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        folder_path = os.path.join(self.split_dir, video_folder)

        frames, frame_names = self._load_frames(folder_path)
        if frames is None:
            return None

        # enforce expected frames
        if frames.shape[0] != self.expected_frames:
            return None

        labels_dict = self._find_labels(video_folder)
        if labels_dict is None:
            return None

        labels = torch.tensor(
            [labels_dict["B"], labels_dict["E"], labels_dict["C"], labels_dict["F"]],
            dtype=torch.long
        )

        out = [frames, labels]

        if self.include_au:
            out.append(self._get_au_features(video_folder, frame_names))
        if self.include_va:
            out.append(self._get_va_features(video_folder, frame_names))

        return tuple(out)


# ------------------------
# Pair dataset wrapper
# ------------------------
def _parse_clip_key(clip_name: str, suffix_digits: int = 2):
    """
    Assumption (same as your examples):
    clip like 1100011002 => video_key=11000110, seg=02
    If your suffix is 3 digits, set suffix_digits=3.
    """
    stem = os.path.splitext(os.path.basename(str(clip_name)))[0]
    if len(stem) <= suffix_digits:
        return stem, 0
    seg = int(stem[-suffix_digits:])
    video_key = stem[:-suffix_digits]
    return video_key, seg


class DAiSEEPairDataset(Dataset):
    """
    Returns pairs sampled within the same video_key.

    Output tuple:
      (frames_i, frames_j, yi, yj, r, w, mask, [au_i, au_j], [va_i, va_j])

    yi,yj: (4,) labels in {0,1,2,3}
    r,w,mask: (4,)
      - mask_k = 1 if |yi_k - yj_k| >= gap_min and yi_k != yj_k else 0
      - r_k = +1 if yi_k > yj_k else -1
      - w_k = |yi_k - yj_k|
    """
    def __init__(self, base_dataset: DAiSEEDataset, gap_min=2, pairs_per_clip=4, suffix_digits=2):
        self.base = base_dataset
        self.gap_min = int(gap_min)
        self.pairs_per_clip = int(pairs_per_clip)
        self.suffix_digits = int(suffix_digits)

        self.video_to_indices = defaultdict(list)
        for idx, clip in enumerate(self.base.video_folders):
            vk, _ = _parse_clip_key(clip, suffix_digits=self.suffix_digits)
            self.video_to_indices[vk].append(idx)

        # only videos with >=2 clips
        self.valid_anchor_indices = []
        for vk, idxs in self.video_to_indices.items():
            if len(idxs) >= 2:
                self.valid_anchor_indices.extend(idxs)

        self._length = max(1, len(self.valid_anchor_indices) * self.pairs_per_clip)

    def __len__(self):
        return self._length

    def __getitem__(self, _k):
        i = random.choice(self.valid_anchor_indices)
        clip_i = self.base.video_folders[i]
        vk, _ = _parse_clip_key(clip_i, suffix_digits=self.suffix_digits)
        idxs = self.video_to_indices[vk]

        j = random.choice(idxs)
        if j == i:
            j = random.choice(idxs)

        item_i = self.base[i]
        item_j = self.base[j]
        if item_i is None or item_j is None:
            return None

        frames_i, yi = item_i[0], item_i[1]
        frames_j, yj = item_j[0], item_j[1]

        diff = (yi - yj).abs()
        mask = (diff >= self.gap_min) & (yi != yj)

        r = torch.where(yi > yj, torch.ones_like(yi), -torch.ones_like(yi)).float()
        w = diff.float()
        mask = mask.float()

        out = [frames_i, frames_j, yi, yj, r, w, mask]

        # optional AU/VA pair
        # item length: 2,3,4
        if len(item_i) >= 3:
            out.append(item_i[2])  # au_i
            out.append(item_j[2])  # au_j
        if len(item_i) == 4:
            out.append(item_i[3])  # va_i
            out.append(item_j[3])  # va_j

        return tuple(out)


def pair_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def get_pair_dataloader(
    split,
    root_dir,
    label_file,
    batch_size=4,
    shuffle=True,
    include_au=False,
    include_va=False,
    au_root=None,
    va_root=None,
    gap_min=2,
    pairs_per_clip=4,
    suffix_digits=2,
    num_workers=4,
):
    base = DAiSEEDataset(
        split=split,
        root_dir=root_dir,
        label_file=label_file,
        au_root=au_root,
        va_root=va_root,
        include_au=include_au,
        include_va=include_va,
        transform_=transform,
        expected_frames=60,
        img_size=224,
    )
    pair_ds = DAiSEEPairDataset(
        base_dataset=base,
        gap_min=gap_min,
        pairs_per_clip=pairs_per_clip,
        suffix_digits=suffix_digits,
    )
    return DataLoader(
        pair_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pair_collate_fn,
        pin_memory=True,
    )