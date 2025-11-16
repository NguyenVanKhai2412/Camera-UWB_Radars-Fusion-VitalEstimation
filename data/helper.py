import sys, time, os
sys.path.append('./loader')
from torch.utils.data import DataLoader
from dataloader import get_data, MultimodalVitalDataset

# Data Loader
start_time = time.time()
data_dir = '../h5_dataset'
data_files = []
for file_name in sorted(os.listdir(data_dir)):
    if '_0.5_' in file_name and '_ideal' in file_name:
        data_files.append(file_name)

rgb_train, novelda_train, umain_train, dca_train, ppg_train, rgb_valid, novelda_valid, umain_valid, dca_valid, ppg_valid = get_data(data_files, data_dir)

dataset_train = MultimodalVitalDataset(rgb_train, dca_train, novelda_train, umain_train, ppg_train)
dataset_valid = MultimodalVitalDataset(rgb_valid, dca_valid, novelda_valid, umain_valid, ppg_valid)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
(inputs, targets) = next(iter(dataloader_train))

print(f"RGB train Shape: {rgb_train.shape}")
print(f"Novelda train Shape: {novelda_train.shape}")
print(f"UMain train Shape: {umain_train.shape}")
print(f"DCA train Shape: {dca_train.shape}")
print(f"PPG train Shape: {ppg_train.shape}")
print('-' * 24)
print(f"RGB valid Shape: {rgb_valid.shape}")
print(f"Novelda valid Shape: {novelda_valid.shape}")
print(f"UMain valid Shape: {umain_valid.shape}")
print(f"DCA valid Shape: {dca_valid.shape}")
print(f"PPG valid Shape: {ppg_valid.shape}")
print('-' * 24)
print(f"RGB Batch Shape: {inputs[0].shape} | {inputs[0].dtype}")
print(f"DCA Batch Shape: {inputs[1].shape} | {inputs[1].dtype}")
print(f"PPG Batch Shape: {targets.shape} | {targets.dtype}")
print('-' * 24)
print(f'Load data time: {(time.time() - start_time)/60:.1f} mins')

print()
print('#' * 24, end='\n\n\n')