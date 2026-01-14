import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from dataloader import get_dataloader
from model_inception import MultiTaskVideoClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Infer attention weights before classifier and save")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for dataset")
    parser.add_argument("--label_folder", type=str, required=True, help="Path to label CSV folder")
    parser.add_argument("--split", type=str, default="Test", choices=["Train", "Validation", "Test"], help="Dataset split")
    parser.add_argument("--label_file", type=str, default=None, help="Optional explicit CSV path; overrides label_folder+*Labels.csv if provided")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save attention weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size (1 to align names)")
    return parser.parse_args()


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Decide label file
    if args.label_file is None:
        label_map = {
            "Train": "TrainLabels.csv",
            "Validation": "ValidationLabels.csv",
            "Test": "AllLabels.csv"  # keep current project convention
        }
        label_file = os.path.join(args.label_folder, label_map[args.split])
    else:
        label_file = args.label_file

    # Use deterministic order so dataset.video_folders aligns with loader
    dataloader = get_dataloader(args.split, args.data_root, label_file, batch_size=args.batch_size, shuffle=False)
    dataset = dataloader.dataset

    model = MultiTaskVideoClassifier().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)

    sample_idx = 0
    for batch in tqdm(dataloader, desc=f"Inferring attention on {args.split}"):
        if batch is None:
            sample_idx += 1
            continue
        frames, labels = batch
        frames = frames.to(device)

        # frames: (B, S, 3, H, W)
        B, S, _, _, _ = frames.shape

        # Forward step-by-step to access features before classifier attention
        # Note: model parts exist as attributes in the composed model
        bsz = B
        seq_len = S
        x = frames
        x = x.view(bsz * seq_len, 3, frames.shape[-2], frames.shape[-1])
        feats = model.feature_extractor(x)                    # (B*S, 2048)
        feats = feats.view(bsz, seq_len, -1)                  # (B, S, 2048)
        feats = model.projection(feats)                       # (B, S, 1024)
        feats = model.transformer(feats)                      # (B, S, 1024)

        # Compute attention weights used by classifier (shared for 4 heads)
        attn_logits = model.classifier.attention_weights(feats)  # (B, S, 1)
        attn_weights = softmax(attn_logits)                      # (B, S, 1)
        attn_weights_np = attn_weights.squeeze(-1).detach().cpu().numpy()  # (B, S)

        # Also get predictions if useful
        logits_list = model.classifier(feats)  # list of 4 tensors (B, num_classes)
        preds = [torch.argmax(l, dim=1).detach().cpu().numpy() for l in logits_list]

        # Save per-sample
        for i in range(B):
            # Try to infer a stable name: if batch_size==1, use dataset.video_folders[sample_idx]
            if args.batch_size == 1 and sample_idx < len(dataset.video_folders):
                vid_name = str(dataset.video_folders[sample_idx])
            else:
                vid_name = f"sample_{sample_idx:06d}"

            out_base = os.path.join(args.output_dir, vid_name)
            np.save(out_base + "_attn.npy", attn_weights_np[i])

            # Save predictions as small csv-like txt
            with open(out_base + "_pred.txt", "w") as f:
                f.write(
                    f"pred_B:{int(preds[0][i])}, pred_E:{int(preds[1][i])}, pred_C:{int(preds[2][i])}, pred_F:{int(preds[3][i])}\n"
                )

            sample_idx += 1

    print(f"Saved attention weights to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)


