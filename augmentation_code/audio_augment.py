import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Gain
import random
seed = 42  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# List of single augmentations
augmentations = [
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
    PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
    Gain(min_gain_in_db=-3, max_gain_in_db=3, p=1.0),
]

def augment_and_save_separate(input_path, output_dir, num_augments=3, sample_rate=16000):
    y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i in range(num_augments):
        # Pick a random augmentation (different each time)
        aug = random.choice(augmentations, replace=False)

        # Apply the augmentation
        y_aug = aug(samples=y, sample_rate=sr)

        out_fname = f"aug{i+1}_{aug.__class__.__name__}_{base_name}.wav"
        out_path = os.path.join(output_dir, out_fname)
        sf.write(out_path, y_aug, sr)
        print(f"Saved: {out_path}")

# Paths
input_dir = "./preprocessed_data/audio/train"
output_dir = "./augmented_data/audio"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".wav"):
        file_path = os.path.join(input_dir, fname)
        augment_and_save_separate(file_path, output_dir, num_augments=3)
