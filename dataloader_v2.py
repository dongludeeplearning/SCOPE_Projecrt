import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import re
import json
import numpy as np
from torchvision import transforms

class DAiSEEDataset(Dataset):
    def __init__(self, split, root_dir, label_file, au_root=None, va_root=None, include_au=False, include_va=False, transform=None):
        """
        Initializes the dataset by loading video folders and labels.
        """
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.labels = pd.read_csv(label_file)
        self.labels.set_index('ClipID', inplace=True)
        
        # Check if split_dir exists, if not, try with lowercase or handle gracefully
        if not os.path.exists(self.split_dir):
            print(f"Warning: Directory {self.split_dir} does not exist. Please check paths.")
            self.video_folders = []
        else:
            self.video_folders = sorted([f for f in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, f))])

        self.transform = transform
        self.include_au = include_au
        self.include_va = include_va
        self.au_root = au_root
        self.va_root = va_root
        
        # Prepare AU/VA dirs if needed
        if self.include_au and self.au_root:
            self.au_split_dir = os.path.join(self.au_root, split)
            if not os.path.exists(self.au_split_dir):
                print(f"Warning: AU Split dir {self.au_split_dir} not found.")

        if self.include_va and self.va_root:
            # Based on previous exploration: data/DAiSEE_AU_VA/Output/Validation
            # The structure is Output/<Split> or Output/Validation?
            # Assuming standard naming: split argument is "Train", "Test", "Validation"
            self.va_split_dir = os.path.join(self.va_root, split) 
            if not os.path.exists(self.va_split_dir):
                print(f"Warning: VA Split dir {self.va_split_dir} not found. Trying 'Validation' if split is 'Validation'...")
                # Fallback check handled during access or let it fail gently

    def __len__(self):
        return len(self.video_folders)

    def parse_au_string(self, au_string):
        """Parses string like 'AU1: 47.38 ...' into a list of floats (41 dims)."""
        values = re.findall(r':\s*(\d+\.\d+)', au_string)
        if not values:
             values = re.findall(r'(\d+\.\d+)', au_string)
        return [float(v) for v in values]

    def get_au_features(self, video_id, frame_names):
        """Loads AU features for a specific video."""
        if not self.au_root:
            return torch.zeros((len(frame_names), 41))
            
        json_path = os.path.join(self.au_split_dir, video_id, f"{video_id}.json")
        features_list = []
        
        if not os.path.exists(json_path):
            return torch.zeros((len(frame_names), 41))

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for frame_name in frame_names:
                if frame_name in data:
                    item = data[frame_name]
                    if "Probabilities" in item and len(item["Probabilities"]) > 0:
                        raw_str = item["Probabilities"][0]
                        vec = self.parse_au_string(raw_str)
                        if len(vec) != 41:
                            curr_len = len(vec)
                            if curr_len < 41:
                                vec = vec + [0.0] * (41 - curr_len)
                            else:
                                vec = vec[:41]
                        features_list.append(vec)
                    else:
                        features_list.append([0.0] * 41)
                else:
                    features_list.append([0.0] * 41)
        except Exception:
            return torch.zeros((len(frame_names), 41))

        return torch.tensor(features_list, dtype=torch.float32)

    def get_va_features(self, video_id, frame_names):
        """Loads VA features for a specific video."""
        if not self.va_root:
            return torch.zeros((len(frame_names), 2))

        video_dir = os.path.join(self.va_split_dir, video_id)
        features_list = []
        
        if not os.path.exists(video_dir):
            return torch.zeros((len(frame_names), 2))

        for frame_name in frame_names:
            json_name = frame_name.replace('.jpg', '.json').replace('.png', '.json')
            json_path = os.path.join(video_dir, json_name)
            val, aro = 0.0, 0.0
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'Valence' in data: val = float(data['Valence'])
                        elif 'valence' in data: val = float(data['valence'])
                        
                        if 'Arousal' in data: aro = float(data['Arousal'])
                        elif 'arousal' in data: aro = float(data['arousal'])
                except:
                    pass
            features_list.append([val, aro])

        return torch.tensor(features_list, dtype=torch.float32)

    def load_frames(self, folder_path):
        """Loads frames from folder."""
        if not os.path.exists(folder_path):
            return None, []
        
        frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        if len(frame_files) == 0:
            return None, []
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        if len(frames) == 0:
            return None, []

        return torch.stack(frames), frame_files

    def find_labels(self, video_id):
        """Finds labels for video."""
        normalized_index = {str(idx).strip().lower(): idx for idx in self.labels.index}
        candidates = [str(video_id), f"{video_id}.avi", f"{video_id}.mp4", f"{video_id}.mkv"]
        
        resolved_key = None
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in normalized_index:
                resolved_key = normalized_index[key]
                break

        if resolved_key is None:
            return None

        row = self.labels.loc[resolved_key, ['Boredom', 'Engagement', 'Confusion', 'Frustration']]
        return {'B': row['Boredom'], 'E': row['Engagement'], 'C': row['Confusion'], 'F': row['Frustration']}

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        folder_path = os.path.join(self.split_dir, video_folder)
        
        frames, frame_names = self.load_frames(folder_path)
        labels_dict = self.find_labels(video_folder)
        
        if labels_dict is None or frames is None:
            return None

        labels = torch.tensor([labels_dict['B'], labels_dict['E'], labels_dict['C'], labels_dict['F']], dtype=torch.long)
        
        result = [frames, labels]

        if self.include_au:
            au_features = self.get_au_features(video_folder, frame_names)
            result.append(au_features)
        
        if self.include_va:
            va_features = self.get_va_features(video_folder, frame_names)
            result.append(va_features)
        
        return tuple(result)

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    
    # Check frame count (assuming 60) for first element of tuple
    batch = [sample for sample in batch if sample[0] is not None and sample[0].shape[0] == 60]
    if len(batch) == 0:
        return None

    return torch.utils.data.default_collate(batch)

def get_dataloader(split, root_dir, label_file, batch_size=16, shuffle=True, include_au=False, include_va=False, au_root=None, va_root=None):
    dataset = DAiSEEDataset(split, root_dir, label_file, au_root=au_root, va_root=va_root, include_au=include_au, include_va=include_va, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn, pin_memory=True)

if __name__ == "__main__":
    # Path Configuration
    ROOT_DATASET = "/Work/Datasets/DAiSEE_Process/DataSet"
    LABEL_FILE = "/mnt/pub/Cognitive/DAiSEE/Labels/TestLabels.csv"
    AU_ROOT = "/Work/Research6/Cog_Detection/DAiSEE_reproduce/data/DAiSEE_AU_VA/DAiSEE_AU"
    VA_ROOT = "/Work/Research6/Cog_Detection/DAiSEE_reproduce/data/DAiSEE_AU_VA/OutpuDAiSEE_VA"
    
    # Update SPLIT to match folder structure (Test, Validation, Train)
    SPLIT = "Test" 
    
    print("--- Testing AU Mode ---")
    loader_au = get_dataloader(SPLIT, ROOT_DATASET, LABEL_FILE, batch_size=2, shuffle=False, 
                               include_au=True, include_va=False, au_root=AU_ROOT, va_root=VA_ROOT)
    
    for i, batch in enumerate(loader_au):
        if batch is None: continue
        if len(batch) == 3:
            fr, lbl, aus = batch
            from pdb import set_trace; set_trace()
            print(f"Batch {i} (AU): Frames {fr.shape}, Labels {lbl.shape}, AU {aus.shape}")
        else:
            print(f"Batch {i} (AU): Unexpected length {len(batch)}")
        break

    print("\n--- Testing AU + VA Mode ---")
    loader_au_va = get_dataloader(SPLIT, ROOT_DATASET, LABEL_FILE, batch_size=2, shuffle=False, 
                                  include_au=True, include_va=True, au_root=AU_ROOT, va_root=VA_ROOT)
    
    for i, batch in enumerate(loader_au_va):
        if batch is None: continue
        if len(batch) == 4:
            fr, lbl, aus, vas = batch
            print(f"Batch {i} (AU+VA): Frames {fr.shape}, Labels {lbl.shape}, AU {aus.shape}, VA {vas.shape}")
        else:
             print(f"Batch {i} (AU+VA): Unexpected length {len(batch)}")
        break
