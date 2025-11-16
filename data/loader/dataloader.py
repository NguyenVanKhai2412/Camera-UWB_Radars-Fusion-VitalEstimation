import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def get_data(filenames, data_dir):
    # Store train/valid data in lists first
    train_data = {'rgb': [], 'novelda': [], 'umain': [], 'dca': [], 'ppg': []}
    valid_data = {'rgb': [], 'novelda': [], 'umain': [], 'dca': [], 'ppg': []}

    for filename in filenames:
        file_path = os.path.join(data_dir, filename)
        with h5py.File(file_path, 'r') as hf:
            # Read entire dataset once
            data_dict = {key: np.array(hf[key]) for key in train_data.keys()}

            # i = np.random.randint(5)
            i = 0

            for key in train_data.keys():
                train_data[key].append(data_dict[key][i][:int(0.8*data_dict[key].shape[1])])

            for key in valid_data.keys():
                valid_data[key].append(data_dict[key][i][int(0.8*data_dict[key].shape[1]):])

    # Convert lists to numpy arrays (concatenating only once for efficiency)
    for key in train_data:
        train_data[key] = np.concatenate(train_data[key], axis=0)
        valid_data[key] = np.concatenate(valid_data[key], axis=0)

    # Compute train mean/std values **once** for normalization
    train_min = {key: np.min(train_data[key], axis=0, keepdims=True) for key in train_data}
    train_max = {key: np.max(train_data[key], axis=0, keepdims=True) for key in train_data}

    # Apply Min-Max Scaling (avoid division by zero)
    for key in train_data:
        denom = train_max[key] - train_min[key]
        denom[denom == 0] = 1  # Prevent division by zero
        train_data[key] = (train_data[key] - train_min[key]) / denom

    for key in valid_data:
        denom = train_max[key] - train_min[key]
        denom[denom == 0] = 1  # Prevent division by zero
        valid_data[key] = (valid_data[key] - train_min[key]) / denom  # Use train min/max

    return (
        train_data['rgb'], train_data['novelda'], train_data['umain'], train_data['dca'], train_data['ppg'],
        valid_data['rgb'], valid_data['novelda'], valid_data['umain'], valid_data['dca'], valid_data['ppg']
    )


class MultimodalVitalDataset(Dataset):
    def __init__(self, rgb_data, dca_data, novelda_data, umain_data, ppg_data):
        self.rgb_data = rgb_data
        self.novelda_data = novelda_data
        self.umain_data = umain_data
        self.dca_data = dca_data
        self.ppg_data = ppg_data

    def __len__(self):
        return len(self.ppg_data)

    def __getitem__(self, idx):
        rgb = torch.tensor(self.rgb_data[idx], dtype=torch.float32).permute(3, 0, 1, 2)     # (3, 100, 80, 80)
        novelda = torch.tensor(self.novelda_data[idx], dtype=torch.float32)                 # (100, 90)
        umain = torch.tensor(self.umain_data[idx], dtype=torch.float32)                     # (100, 30)
        dca = torch.tensor(self.dca_data[idx], dtype=torch.float32)                         # (8, 100, 80)
        ppg = torch.tensor(self.ppg_data[idx], dtype=torch.float32)                         # (100,)

        return (rgb, dca, novelda, umain), ppg  # Inputs and Ground Truth