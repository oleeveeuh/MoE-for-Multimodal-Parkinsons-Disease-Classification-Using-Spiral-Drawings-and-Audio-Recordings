from PIL import Image
import torchvision.transforms as transforms
import os
import shutil
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold



# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    # transforms.CenterCrop((224, 224)),            # Crop to center region
    transforms.Resize((224, 224)),                # Resize to 224x224
    transforms.ToTensor(),                        # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize grayscale image
])

# Function to apply transformations and save
def preprocess_and_save(image_path, save_path):
    image = Image.open(image_path).convert('RGB')  # Ensure RGB input
    image = preprocess(image)                      # Apply transforms
    # Convert back to PIL image for saving (optional)
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image)
    pil_image.save(save_path)


processed_images = []
labels = []
path = "./KaggleSpiral/spiral/alldata/"
save_path = "./preprocessed_data/image/"
os.makedirs(save_path, exist_ok=True)
healthy_path = os.path.join(path, 'healthy')
pd_path = os.path.join(path, 'parkinson')

for file in os.listdir(healthy_path):
    file_path = os.path.join(healthy_path, file)

    new_save_path = os.path.join(save_path, ('H_' + file))
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    if file.lower().endswith(('.png')):
        processed_images.append(new_save_path)
        labels.append(0)
        preprocess_and_save(file_path,new_save_path)

for file in os.listdir(pd_path):
    file_path = os.path.join(pd_path, file)
    new_save_path = os.path.join(save_path, ('PD_' + file))
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
      
    if file.lower().endswith(('.png')):
        processed_images.append(new_save_path)
        labels.append(1)
        preprocess_and_save(file_path, new_save_path)

X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42, stratify=labels)


for path in X_train:
    new_save_path = "./preprocessed_data/image/train"
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    shutil.copy(path,new_save_path)


for path in X_test:
    new_save_path = "./preprocessed_data/image/test"
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    shutil.copy(path,new_save_path)


