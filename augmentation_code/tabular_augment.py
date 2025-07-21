import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd

import torch.nn.functional as F
torch.manual_seed(42)
np.random.seed(42)

def load_data_from_file(file_path):
    """
    Loads data from a given file (for example, CSV or text file).
    """
    features = []

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and split by ';'
            data = line.strip().split(';')

            # Parse values from the line
            x = float(data[0].strip())   # X coordinate
            y = float(data[1].strip())   # Y coordinate
            z = float(data[2].strip())   # Z coordinate
            pressure = float(data[3].strip())  # Pressure
            timestamp = float(data[4].strip())  # Timestamp

            features.append([x, y, z, pressure, timestamp])

    return torch.tensor(features, dtype=torch.float32)

def mixup_data(x1, x2, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)  # Sample lambda from Beta distribution
    return lam * x1 + (1 - lam) * x2, lam

def perform_mixup_on_random_files(directory, alpha=1.0):
    '''Select two random files from the directory and perform Mixup on their data'''
    # List all files in the directory
    files = os.listdir(directory)
    save_dir = "./augmented_data/tabular"

    mixed_x = []
    # Select two random files
    file1, file2 = np.random.choice(files, 2, replace=True)
    
    if file1[0] != file2[0]:
        return None
    
    if aug_count < 18:
        if file1[0] == 'H':
            return None 
    else: 
        if file1[0] == 'P':
            return None 


    # Load data from the two files
    file1_path = os.path.join(directory, file1)
    file2_path = os.path.join(directory, file2)
    
    i, j = os.path.basename(file1_path).split('.')[0], os.path.basename(file2_path).split('.')[0]
    out_name = (f"mix_{i}_{j}")
    save_path = os.path.join(save_dir, out_name + '.txt')
    cnt = 0

    while os.path.exists(save_path):
        new_filename = f"{out_name}_{cnt}"  # e.g., mix_Healthy(10)_Healthy(8)_1.txt
        save_path = os.path.join(save_dir, new_filename+ '.txt')
        cnt+=1

    x1 = load_data_from_file(file1_path)
    x2 = load_data_from_file(file2_path)

    x1_aligned, x2_aligned, mask1, mask2 = check_and_align_shapes(x1, x2, max_length_diff=300)

    if x1_aligned is not None:
        mixed_x, lam = mixup_data(x1_aligned, x2_aligned, alpha)

        # Combine masks: weighted average (or just union)
        mixed_mask = mask1 * lam + mask2 * (1 - lam)

        np.savetxt(save_path, mixed_x, delimiter=';', fmt='%.4f')
        np.savetxt(save_path.replace('mix', 'mask'), mixed_mask, delimiter=';', fmt='%.1f')
    else:
        print(f"Rejected due to shape difference of {len(x1)-len(x2)}")
        return None
    
    return mixed_x

def check_and_align_shapes(x1, x2, max_length_diff=50):
    len1, len2 = x1.shape[0], x2.shape[0]
    
    if abs(len1 - len2) > max_length_diff:
        return None, None, None, None

    # Determine which file is shorter
    if len1 < len2:
        x1_padded = np.pad(x1, ((0, len2 - len1), (0, 0)), mode='constant')
        x2_padded = x2.clone()
        mask1 = np.zeros((len2,), dtype=np.float32)
        mask2 = np.ones((len2,), dtype=np.float32)
        mask1[:len1] = 1.0
    elif len2 < len1:
        x2_padded = np.pad(x2, ((0, len1 - len2), (0, 0)), mode='constant')
        x1_padded = x1.clone()
        mask1 = np.ones((len1,), dtype=np.float32)
        mask2 = np.zeros((len1,), dtype=np.float32)
        mask2[:len2] = 1.0
    else:
        x1_padded = x1
        x2_padded = x2
        mask1 = np.ones((len1,), dtype=np.float32)
        mask2 = np.ones((len2,), dtype=np.float32)

    return torch.tensor(x1_padded, dtype=torch.float32), torch.tensor(x2_padded, dtype=torch.float32), mask1, mask2

def second_mixup_on_random_files(directory1, directory2, index, alpha=1.0):
    '''Select two random files from the directory and perform Mixup on their data'''
    # List all files in the directory
    save_dir = "./augmented_data/tabular"

    mixed_x = []
    # Select two random files
    file1 = np.random.choice(os.listdir(directory1), 1, replace=True)[0]
    file2 = np.random.choice(os.listdir(directory2), 1, replace=True)[0]
    
    if index == 4:
        if not (file2.startswith("mix_")):
            return None

    if index == 5:
        if not (file2.startswith("mix3_")):
            return None
    
    if file1[0] != file2[index]:
        return None
    
    if aug_count < 18:
        if file1[0] == 'H':
            return None 
    else: 
        if file1[0] == 'P':
            return None 


    # Load data from the two files
    file1_path = os.path.join(directory1, file1)
    file2_path = os.path.join(directory2, file2)
    
    i, j = os.path.basename(file1_path).split('.')[0], os.path.basename(file2_path).split('.')[0]
    out_name = (f"mix{index-1}_{i}_{j}")
    save_path = os.path.join(save_dir, out_name + '.txt')
    cnt = 0

    while os.path.exists(save_path):
        new_filename = f"{out_name}_{cnt}"  # e.g., mix_Healthy(10)_Healthy(8)_1.txt
        save_path = os.path.join(save_dir, new_filename+ '.txt')
        cnt+=1

    x1 = load_data_from_file(file1_path)
    x2 = load_data_from_file(file2_path)

    x1_aligned, x2_aligned, mask1, mask2 = check_and_align_shapes(x1, x2, max_length_diff=300)

    if x1_aligned is not None:
        mixed_x, lam = mixup_data(x1_aligned, x2_aligned, alpha)

        # Combine masks: weighted average (or just union)
        mixed_mask = mask1 * lam + mask2 * (1 - lam)

        np.savetxt(save_path, mixed_x, delimiter=';', fmt='%.4f')
        np.savetxt(save_path.replace('mix', 'mask'), mixed_mask, delimiter=';', fmt='%.1f')
    else:
        print(f"Rejected due to shape difference of {len(x1)-len(x2)}")
        return None
    
    return mixed_x

train_path = "./preprocessed_data/tabular/train"
aug_count = 0
# test_path = "./preprocessed_data/tabular/test"

# while aug_count < 43: #18+25 
#     mixed_x = perform_mixup_on_random_files(train_path, alpha=1.0)
#     if mixed_x != None:
#         aug_count +=1

# aug_count = 0
# while aug_count < 43:
#     mixed_x = second_mixup_on_random_files(train_path, "./augmented_data/tabular", index = 4, alpha=1.0)
#     if mixed_x != None:
#         aug_count +=1

aug_count = 0
while aug_count < 43:
    mixed_x = second_mixup_on_random_files(train_path, "./augmented_data/tabular", index = 5, alpha=1.0)
    if mixed_x != None:
        aug_count +=1



