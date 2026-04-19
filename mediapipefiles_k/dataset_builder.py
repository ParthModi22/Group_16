import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List

# Same normalization function from Phase 1 so we can use it dynamically on any data
def normalize_landmarks(obs):
    r_hip_idx, l_hip_idx = 23 * 4, 24 * 4
    r_sho_idx, l_sho_idx = 12 * 4, 11 * 4
    
    mid_hip_x = (obs[:, r_hip_idx] + obs[:, l_hip_idx]) / 2.0
    mid_hip_y = (obs[:, r_hip_idx+1] + obs[:, l_hip_idx+1]) / 2.0
    mid_hip_z = (obs[:, r_hip_idx+2] + obs[:, l_hip_idx+2]) / 2.0
    
    mid_sho_x = (obs[:, r_sho_idx] + obs[:, l_sho_idx]) / 2.0
    mid_sho_y = (obs[:, r_sho_idx+1] + obs[:, l_sho_idx+1]) / 2.0
    mid_sho_z = (obs[:, r_sho_idx+2] + obs[:, l_sho_idx+2]) / 2.0
    
    torso_height = np.sqrt((mid_sho_x - mid_hip_x)**2 + (mid_sho_y - mid_hip_y)**2 + (mid_sho_z - mid_hip_z)**2)
    torso_height = np.where(torso_height == 0, 1.0, torso_height)
    
    norm_obs = np.copy(obs)
    for i in range(33):
        idx = i * 4
        norm_obs[:, idx]   = (obs[:, idx] - mid_hip_x) / torso_height
        norm_obs[:, idx+1] = (obs[:, idx+1] - mid_hip_y) / torso_height
        norm_obs[:, idx+2] = (obs[:, idx+2] - mid_hip_z) / torso_height
    return norm_obs

class GestureSequenceDataset(Dataset):
    def __init__(self, file_paths: List[str], sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.features = []
        self.labels = []
        
        for label_idx, file_path in enumerate(file_paths):
            try:
                df = pd.read_csv(file_path)
                df.fillna(0, inplace=True)
                data = df.values[:, :132] 
                
                # Normalize
                data = normalize_landmarks(data).astype(np.float32)
                
                # Sliding window
                num_frames = data.shape[0]
                for i in range(num_frames - self.sequence_length + 1):
                    window = data[i : i + self.sequence_length, :]
                    self.features.append(window)
                    self.labels.append(label_idx)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # ensure labels are converted properly
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx]).long()

if __name__ == "__main__":
    files = ["dataset_clap.csv", "dataset_disco.csv", "dataset_hello.csv", "dataset_wakanda.csv", "dataset_zombie.csv"]
    dataset = GestureSequenceDataset(files, sequence_length=30)
    print(f"Dataset created with {len(dataset)} sequences of length 30.")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch_features, batch_labels = next(iter(loader))
    print(f"Batch shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
