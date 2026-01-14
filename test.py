import os
import time
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataloader import get_dataloader
from tqdm import tqdm

# Dynamic Model Import
def get_model(model_type, device, **kwargs):
    if model_type == "v1":
        from model_inception_v1 import MultiTaskVideoClassifier as ModelV1
        return ModelV1(device=device) 
    elif model_type == "v2":
        from model_inception_v2 import MultiTaskVideoClassifier as ModelV2
        # Force enable Positional Encoding for v2 to match training config
        return ModelV2(device=device, use_pos_encoding=True, **kwargs)
    elif model_type == "v3":
        from model_inception_v3 import MultiTaskVideoClassifier as ModelV3
        return ModelV3(device=device, use_pos_encoding=True, **kwargs)
    elif model_type == "v4":
        from model_inception_v4 import MultiTaskVideoClassifier as ModelV4
        return ModelV4(device=device, use_pos_encoding=True, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Test Multi-Task Video Classifier on DAiSEE")
    parser.add_argument("--model_type", type=str, default="v1", choices=["v1", "v2", "v3", "v4"], help="Model version to test")
    parser.add_argument("--data_root", type=str, default="/mnt/pub/Cognitive/DAiSEE_Process/DataSet", help="Root directory for dataset")
    parser.add_argument("--label_folder", type=str, default="/mnt/pub/Cognitive/DAiSEE/Labels", help="Path to label CSV file")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for testing")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained model checkpoint (if None, infers from model_type)")
    parser.add_argument("--log_file", type=str, default=None, help="File to store test logs (if None, infers from model_type)")
    return parser.parse_args()

# Compute Top-1 Accuracy
def compute_accuracy(outputs, labels):
    accuracies = []
    for i in range(4):
        preds = torch.argmax(outputs[i], dim=1)
        acc = (preds == labels[:, i]).float().mean().item()
        accuracies.append(acc)
    return accuracies

# Log Function
def log_print(message, log_file):
    print(message, flush=True)  # Display in real-time
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()  # Write to file in real-time

# Test Function
def test_model(model, test_loader, device, log_file):
    model.eval()
    total_acc = [0, 0, 0, 0]
    num_batches = 0

    with torch.no_grad():
        for batch_frames, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            outputs = model(batch_frames)
            acc = compute_accuracy(outputs, batch_labels)
            total_acc = [total_acc[i] + acc[i] for i in range(4)]
            num_batches += 1

    avg_acc = [a / num_batches for a in total_acc]
    log_print(f"Test Accuracy: {avg_acc}", log_file)
    log_print(f"Task B (Boredom) Accuracy: {avg_acc[0]:.4f}", log_file)
    log_print(f"Task E (Engagement) Accuracy: {avg_acc[1]:.4f}", log_file)
    log_print(f"Task C (Confusion) Accuracy: {avg_acc[2]:.4f}", log_file)
    log_print(f"Task F (Frustration) Accuracy: {avg_acc[3]:.4f}", log_file)
    log_print(f"Average Accuracy: {sum(avg_acc)/len(avg_acc):.4f}", log_file)

# Main Testing Loop
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader("Test", args.data_root, os.path.join(args.label_folder, "AllLabels.csv"), batch_size=args.batch_size)
    
    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_filename = f"inception-transformer-{args.model_type}.pth"
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
    else:
        checkpoint_path = args.checkpoint
    
    # Determine log file path
    if args.log_file is None:
        args.log_file = os.path.join(args.checkpoint_dir, f"test_log_{args.model_type}.txt")
    
    # Ensure log directory exists
    log_dir = os.path.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"Testing started at {start_time}", args.log_file)
    log_print(f"Model Type: {args.model_type}", args.log_file)
    
    # Initialize model based on type
    try:
        model = get_model(args.model_type, device).to(device)
    except Exception as e:
        print(f"Error initializing model {args.model_type}: {e}")
        return

    # Load Checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log_print(f"Loaded model from {checkpoint_path}", args.log_file)
    else:
        log_print(f"Checkpoint not found at {checkpoint_path}", args.log_file)
        return
    
    test_model(model, test_loader, device, args.log_file)
    
    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"Testing completed at {end_time}", args.log_file)

if __name__ == "__main__":
    main()
