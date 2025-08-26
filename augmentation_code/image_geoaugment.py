import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

def set_seed(seed=42):
    random.seed(seed)

set_seed()

import shutil

def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Custom transform for non-uniform scaling (stretch)
class RandomStretch:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, img):
        w, h = img.size
        sx = random.uniform(*self.scale_range)
        sy = random.uniform(*self.scale_range)
        return img.resize((int(w * sx), int(h * sy)), Image.BILINEAR)

# Define individual augmentations
AUGS = [

    # transforms.RandomRotation(degrees=15),
    # transforms.RandomResizedCrop(224, scale=(0.95, 1.05), ratio=(0.95, 1.05)),  # mild zoom/crop
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    # RandomStretch(),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
        # You can optionally include stretch if analyzed/tested
]

def apply_random_augmentations(img, n):
    """Randomly select and apply `n` different augmentations to the image."""
    chosen = random.sample(AUGS, n)
    for aug in chosen:
        img = aug(img)
    img = transforms.Resize((224, 224))(img)  # Restore shape
    return img

# Paths
input_dir = './preprocessed_data/image/train/'
output_dir = './augmented_data/image/geometric2'
os.makedirs(output_dir, exist_ok=True)
clear_directory(output_dir)


# Augment and save images
for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.png'):
        continue

    img_path = os.path.join(input_dir, fname)
    img = Image.open(img_path).convert('RGB')

    for n in [1, 2]:
        aug_img = apply_random_augmentations(img.copy(), n)
        save_name = f"{os.path.splitext(fname)[0]}_aug{n}.png"
        F.to_pil_image(transforms.ToTensor()(aug_img)).save(os.path.join(output_dir, save_name))
