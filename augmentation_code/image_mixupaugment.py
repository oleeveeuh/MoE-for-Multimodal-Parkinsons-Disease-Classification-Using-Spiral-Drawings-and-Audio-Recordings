import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import os

def load_image_as_tensor(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    else:
        image = transforms.ToTensor()(image)  # Default to [0,1] range tensor
    return image

def mixup_images(image1, image2, alpha=1.0):
    """
    image1, image2: torch.Tensor of shape (C, H, W)
    alpha: Beta distribution parameter
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Ensure shapes match
    assert image1.shape == image2.shape, "Images must have the same shape"
    
    mixed_image = lam * image1 + (1 - lam) * image2
    return mixed_image, lam

def perform_mixup_on_random_files(directory, alpha=1.0):
    '''Select two random files from the directory and perform Mixup on their data'''
    # List all files in the directory
    files = os.listdir(directory)
    save_dir = "./augmented_data/image"

    mixed_x = []
    # Select two random files
    file1, file2 = np.random.choice(files, 2, replace=True)
    
    if file1[0] != file2[0]:
        return None
    
    if aug_count < 41:
        if file1[0] == 'P':
            return None 
    else: 
        if file1[0] == 'H':
            return None 

    file1_path = os.path.join(directory, file1)
    file2_path = os.path.join(directory, file2)
    
    i, j = os.path.basename(file1_path).split('.')[0], os.path.basename(file2_path).split('.')[0]
    out_name = (f"mix_{i}_{j}")
    save_path = os.path.join(save_dir, out_name + '.png')
    cnt = 0

    while os.path.exists(save_path):
        new_filename = f"{out_name}_{cnt}"  # e.g., mix_Healthy(10)_Healthy(8)_1.txt
        save_path = os.path.join(save_dir, new_filename+ '.png')
        cnt+=1

    x1 = load_image_as_tensor(file1_path)
    x2 = load_image_as_tensor(file2_path)

    mixed_img, lam = mixup_images(x1, x2, alpha=0.4)

    to_pil = transforms.ToPILImage()
    mixed_pil = to_pil(mixed_img)
    mixed_pil.save(save_path)
    
    return mixed_img

train_path = "./preprocessed_data/image/train"
aug_count = 0
# test_path = "./preprocessed_data/tabular/test"

while aug_count < 81:
    mixed_x = perform_mixup_on_random_files(train_path, alpha=1.0)
    if mixed_x != None:
        aug_count +=1

