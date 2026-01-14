import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from dataloader import get_dataloader
from tqdm import tqdm
import numpy as np
from pdb import set_trace as st

# Dynamic Model Import
def get_model(model_type, device, **kwargs):
    if model_type == "v1":
        from model_inception_v1 import MultiTaskVideoClassifier as ModelV1
        return ModelV1(device=device)  # Adjust args if needed
    elif model_type == "v2":
        from model_inception_v2 import MultiTaskVideoClassifier as ModelV2
        # Force enable Positional Encoding for v2
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
    parser = argparse.ArgumentParser(description="Train Multi-Task Video Classifier on DAiSEE")
    
    parser.add_argument("--model_type", type=str, default="v1", choices=["v1", "v2", "v3", "v4"], help="Model version to train")
    parser.add_argument("--data_root", type=str, default="/mnt/pub/Cognitive/DAiSEE_Process/DataSet", help="Root directory for dataset")
    parser.add_argument("--label_folder", type=str, default="/mnt/pub/Cognitive/DAiSEE/Labels", help="Path to label CSV file")
    parser.add_argument("--batch_size", type=int, default=4, help="Physical batch size for training (keep small for VRAM)")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="Gradient accumulation steps (Effective Batch Size = batch_size * accumulation_steps)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    # parser.add_argument("--log_file", type=str, default="./checkpoints/log.txt", help="File to store training logs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (number of epochs without improvement)")
    
    return parser.parse_args()

# Log Function
def log_print(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Compute Top-1 Accuracy
def compute_accuracy(outputs, labels):
    accuracies = []
    for i in range(4):
        preds = torch.argmax(outputs[i], dim=1)
        acc = (preds == labels[:, i]).float().mean().item()
        accuracies.append(acc)
    return accuracies

# Train Function
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, log_file, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_acc = [0, 0, 0, 0]
    num_batches = 0
    
    optimizer.zero_grad()  # Initialize gradients once

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Training")):
        # Some batches may be None due to filtering in collate_fn
        if batch is None:
            continue
        batch_frames, batch_labels = batch
        batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)

        # optimizer.zero_grad() # Removed: handled by accumulation logic
        outputs = model(batch_frames)

        # Compute loss
        loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))
        
        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Restore loss for logging (multiply back)
        current_loss = loss.item() * accumulation_steps
        
        # Calculate Accuracy (on the current micro-batch)
        acc = compute_accuracy(outputs, batch_labels)
        total_loss += current_loss
        total_acc = [total_acc[i] + acc[i] for i in range(4)]
        num_batches += 1
        
    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(1, num_batches)
    avg_acc = [a / max(1, num_batches) for a in total_acc]

    log_print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f} - Train Acc: {avg_acc}", log_file)
    return avg_loss, avg_acc

# # Validation Function
# def validate_epoch(model, val_loader, criterion, device, epoch, log_file):
#     model.eval()
#     total_loss = 0
#     total_acc = [0, 0, 0, 0]
#     num_batches = 0

#     with torch.no_grad():
#         for batch_frames, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
#             batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
#             outputs = model(batch_frames)

#             loss = sum(criterion(outputs[i], batch_labels[:, i]) for i in range(4))

#             acc = compute_accuracy(outputs, batch_labels)
#             total_loss += loss.item()
#             total_acc = [total_acc[i] + acc[i] for i in range(4)]
#             num_batches += 1

#     avg_loss = total_loss / num_batches
#     avg_acc = [a / num_batches for a in total_acc]

#     log_print(f"Epoch {epoch} - Val Loss: {avg_loss:.4f} - Val Acc: {avg_acc}", log_file)
#     return avg_loss, avg_acc

# Main Training Loop with Improved Validation Selection
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set log file based on model type and timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"log_{args.model_type}_{timestamp}.txt"
    args.log_file = os.path.join(args.checkpoint_dir, log_filename)


    train_loader = get_dataloader("Train", args.data_root, os.path.join(args.label_folder, "TrainLabels.csv"), batch_size=args.batch_size)
    # val_loader = get_dataloader("Validation", args.data_root, os.path.join(args.label_folder, "ValidationLabels.csv"), batch_size=args.batch_size)


    # Initialize Model based on args.model_type
    print(f"Initializing model: {args.model_type}")
    try:
        kwargs = {}
        # if args.model_type == "v2":
        #     kwargs["some_param"] = True
            
        model = get_model(args.model_type, device, **kwargs).to(device)
    except Exception as e:
        print(f"Error initializing model {args.model_type}: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"Training started at {start_time} with model: {args.model_type}", args.log_file)

    # Validation Optimization Variables
    best_combined_score = float("-inf")  # Track best model
    best_val_loss = float("inf")  # Moving average of validation loss
    patience_counter = 0  # Early stopping counter

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.log_file, args.accumulation_steps)
        # val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch, args.log_file)

        # Compute Combined Validation Score: (Higher is better)
        avg_val_acc = np.mean(train_acc)
        combined_score = avg_val_acc - train_loss  # Balance accuracy & loss

        log_print(f"Epoch {epoch} - Combined Score: {combined_score:.4f}", args.log_file)

        # Save Model Based on Best Score
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_val_loss = train_loss  # Update moving average
            patience_counter = 0  # Reset early stopping

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Save with model type in filename
            checkpoint_filename = f"inception-transformer-{args.model_type}-2nd-try.pth"
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path) 

            log_print(f" Model saved at {checkpoint_path}", args.log_file)
        else:
            patience_counter += 1
            log_print(f"No improvement ({patience_counter}/{args.patience})", args.log_file)

        #  Early Stopping
        if patience_counter >= args.patience:
            log_print(" Early stopping triggered!", args.log_file)
            break

    log_print("Training complete!", args.log_file)

if __name__ == "__main__":
    main()
