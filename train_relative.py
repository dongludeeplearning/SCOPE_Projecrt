# train_relative.py
import os
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from dataloader_relative import get_pair_dataloader
from inception_relative_model import InceptionRelativeRanker


def parse_args():
    p = argparse.ArgumentParser("DAiSEE Pairwise Relative Ranking Training")

    # --- paths (match your previous defaults) ---
    p.add_argument("--data_root", type=str, default="/mnt/pub/Cognitive/DAiSEE_Process/DataSet")
    p.add_argument("--label_folder", type=str, default="/mnt/pub/Cognitive/DAiSEE/Labels")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_relative")

    # --- training ---
    p.add_argument("--batch_size", type=int, default=2)           # physical batch
    p.add_argument("--accumulation_steps", type=int, default=16)  # effective batch
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # --- pair sampling ---
    p.add_argument("--gap_min", type=int, default=2)
    p.add_argument("--pairs_per_clip", type=int, default=4)
    p.add_argument("--suffix_digits", type=int, default=2)  # if your clip suffix is 3 digits, set 3

    # --- aux signals (optional) ---
    p.add_argument("--use_au", action="store_true")
    p.add_argument("--use_va", action="store_true")
    p.add_argument("--au_root", type=str, default="data/DAiSEE_AU_VA/DAiSEE_AU")
    p.add_argument("--va_root", type=str, default="data/DAiSEE_AU_VA/Output")

    # --- model ---
    p.add_argument("--no_pos_encoding", action="store_true")

    return p.parse_args()


def log_print(msg, log_file):
    print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")


def ranking_loss_task(s_i, s_j, r, w, mask):
    """
    s_i,s_j: (B,)
    r,w,mask: (B,)
    loss = w * softplus(-r*(s_i-s_j)), masked average
    """
    margin = r * (s_i - s_j)
    loss = F.softplus(-margin)  # log(1+exp(-margin))
    loss = loss * w * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def train_one_epoch(model, loader, optimizer, device, epoch, log_file, accumulation_steps=1, use_au=False, use_va=False):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        if batch is None:
            continue

        # batch tuple:
        # (fi,fj, yi,yj, r,w,mask, [au_i,au_j], [va_i,va_j])
        fi = batch[0].to(device)
        fj = batch[1].to(device)
        r = batch[4].to(device).float()     # (B,4)
        w = batch[5].to(device).float()     # (B,4)
        m = batch[6].to(device).float()     # (B,4)

        idx = 7
        au_i = au_j = None
        va_i = va_j = None
        if use_au:
            au_i = batch[idx].to(device); au_j = batch[idx+1].to(device); idx += 2
        if use_va:
            va_i = batch[idx].to(device); va_j = batch[idx+1].to(device); idx += 2

        scores_i = model(fi, au=au_i, va=va_i)  # list of 4, each (B,)
        scores_j = model(fj, au=au_j, va=va_j)

        loss = 0.0
        for t in range(4):
            loss = loss + ranking_loss_task(
                scores_i[t].view(-1),
                scores_j[t].view(-1),
                r[:, t].view(-1),
                w[:, t].view(-1),
                m[:, t].view(-1),
            )

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_steps += 1

    if n_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(1, n_steps)
    log_print(f"[Train] Epoch {epoch} | loss={avg_loss:.6f}", log_file)
    return avg_loss


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.checkpoint_dir, f"log_relative_{timestamp}.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Device: {device}", log_file)

    train_label = os.path.join(args.label_folder, "TrainLabels.csv")

    train_loader = get_pair_dataloader(
        split="Train",
        root_dir=args.data_root,
        label_file=train_label,
        batch_size=args.batch_size,
        shuffle=True,
        include_au=args.use_au,
        include_va=args.use_va,
        au_root=args.au_root,
        va_root=args.va_root,
        gap_min=args.gap_min,
        pairs_per_clip=args.pairs_per_clip,
        suffix_digits=args.suffix_digits,
        num_workers=4,
    )

    model = InceptionRelativeRanker(use_pos_encoding=(not args.no_pos_encoding)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    log_print("=== Config ===", log_file)
    log_print(str(vars(args)), log_file)

    best_loss = float("inf")
    ckpt_name = f"inception-relative-gap{args.gap_min}-ppc{args.pairs_per_clip}.pth"
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, log_file,
            accumulation_steps=args.accumulation_steps,
            use_au=args.use_au, use_va=args.use_va
        )

        # save best
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), ckpt_path)
            log_print(f"[CKPT] saved best to {ckpt_path} (loss={best_loss:.6f})", log_file)

    log_print("Training complete.", log_file)


if __name__ == "__main__":
    main()
    