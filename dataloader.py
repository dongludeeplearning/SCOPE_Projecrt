import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
from torchvision import transforms
from pdb import set_trace as st

class DAiSEEDataset(Dataset):
    def __init__(self, split, root_dir, label_file, transform=None):
        """
        Initializes the dataset by loading video folders and labels.

        :param split: The dataset split (e.g., "train", "test")
        :param root_dir: The root directory containing the dataset
        :param label_file: Path to the CSV file containing labels
        :param transform: Transformations to apply to images
        """
        self.split_dir = os.path.join(root_dir, split)
        self.labels = pd.read_csv(label_file)
        # print ("csv column:", self.labels.columns) 
        self.labels.set_index('ClipID', inplace=True)  # Use 'ClipID' as index for easier lookup
        self.video_folders = [f for f in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, f))]
        self.transform = transform

    def __len__(self):
        """Returns the total number of video samples in the dataset."""
        return len(self.video_folders)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        :param idx: Index of the sample
        :return: (Frames as a tensor, Label tensor)
        """
        video_folder = self.video_folders[idx]
        frames = self.load_frames(os.path.join(self.split_dir, video_folder))
        
        # Get labels using the new find_labels function
        labels_dict = self.find_labels(video_folder)
        # If labels missing or frames missing, return None so collate_fn can filter
        if labels_dict is None or frames is None:
            return None

        # Convert label dictionary to tensor
        labels = torch.tensor([labels_dict['B'], labels_dict['E'], labels_dict['C'], labels_dict['F']], dtype=torch.long)
        # print(f" Sample {video_folder}: Frames shape {frames.shape}, Labels shape {labels.shape}")
        return frames, labels

    def load_frames(self, folder_path):
        """
        Loads and processes frames from a given video folder.

        :param folder_path: Path to the folder containing frames
        :return: Tensor of stacked frames
        """
        frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
        if len(frame_files) == 0:
            # print(f"No frames found in {folder_path}!")
            return None  # Return None so we can filter it out in __getitem__
        
        frames = []
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        return torch.stack(frames)

    def find_labels(self, video_id):
        """
        Finds and returns labels for a given video.

        :param video_id: The ID of the video to search for
        :return: A dictionary in the format {'B': x, 'E': y, 'C': z, 'F': w} or None if not found
        """
        # Build normalized index mapping for robust lookup
        normalized_index = {str(idx).strip().lower(): idx for idx in self.labels.index}

        # Try multiple candidate keys (with/without common extensions)
        candidates = [
            str(video_id),
            f"{video_id}.avi",
            f"{video_id}.mp4",
            f"{video_id}.mkv",
        ]

        resolved_key = None
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in normalized_index:
                resolved_key = normalized_index[key]
                break

        if resolved_key is None:
            return None

        row = self.labels.loc[resolved_key, ['Boredom', 'Engagement', 'Confusion', 'Frustration']]
        return {
            'B': row['Boredom'],
            'E': row['Engagement'],
            'C': row['Confusion'],
            'F': row['Frustration']
        }

# Define transformations for frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def collate_fn(batch):
    # Ensure batch is not empty and samples are valid
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:  # If batch is empty, return None to avoid crashing
        print("Warning: Empty batch detected, skipping!")
        return None  

    # Ensure frames have exactly 60 frames before stacking
    batch = [sample for sample in batch if sample[0] is not None and sample[0].shape[0] == 60]

    if len(batch) == 0:
        print("Warning: No valid samples with 60 frames in this batch, skipping!")
        return None  

    return torch.utils.data.default_collate(batch)

def get_dataloader(split, root_dir, label_file, batch_size=16, shuffle=True):
    """
    Creates a DataLoader for the DAiSEE dataset.

    :param split: The dataset split (e.g., "train", "test")
    :param root_dir: The root directory containing the dataset
    :param label_file: Path to the CSV file containing labels
    :param batch_size: Number of samples per batch
    :param shuffle: Whether to shuffle the data
    :return: DataLoader object
    """
    dataset = DAiSEEDataset(split, root_dir, label_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,collate_fn=collate_fn, pin_memory=True)




# Test Cases for dataloader.py
if __name__ == "__main__":
    root_folder = "/Work/Datesets/DAiSEE_Process/DataSet"
    label_folder = "/mnt/pub/CognitiveDataset/DAiSEE/Labels/"
    split = "Test"
    label_path = os.path.join(label_folder, split+"Labels.csv")


    train_loader = get_dataloader("Train", "/mnt/pub/CognitiveDataset/DAiSEE_Process/DataSet", 
                                "/mnt/pub/CognitiveDataset/DAiSEE/Labels/AllLabels.csv", batch_size=32, shuffle= False)

    # Iterate through all batches to check for errors
    for batch_idx, sample in enumerate(train_loader):
        if sample is None:
            print(f"⚠️ Found an empty batch! Batch Index: {batch_idx}")
            break  # Stop at the first error
        
        try:
            frames, labels = sample
            print(f"Batch {batch_idx}: Frames {frames.shape}, Labels {labels.shape}")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")  # Print the exact error message
            break  # Stop at the first error

    # # Test dataset loading
    # dataset = DAiSEEDataset(split, root_folder, label_path, transform=transform)
    # print(f"Dataset size: {len(dataset)}")

    # # Test single item retrieval
    # sample_frames, sample_labels = dataset[0]
    # print(f"Sample frames shape: {sample_frames.shape}")
    # print(f"Sample labels: {sample_labels}")

    # # Test DataLoader
    # dataloader = get_dataloader(split, root_folder, label_path, batch_size=4)
    # batch_frames, batch_labels = next(iter(dataloader))
    
    # print(f"Batch frames shape: {batch_frames.shape}")
    # print(f"Batch labels shape: {batch_labels.shape}")


# csv column: Index(['ClipID', 'Boredom', 'Engagement', 'Confusion', 'Frustration'], dtype='object')
# Dataset size: 1720
# Sample frames shape: torch.Size([60, 3, 224, 224])
# Sample labels: tensor([0, 3, 0, 0])
# csv column: Index(['ClipID', 'Boredom', 'Engagement', 'Confusion', 'Frustration'], dtype='object')
# Batch frames shape: torch.Size([4, 60, 3, 224, 224])
# Batch labels shape: torch.Size([4, 4])