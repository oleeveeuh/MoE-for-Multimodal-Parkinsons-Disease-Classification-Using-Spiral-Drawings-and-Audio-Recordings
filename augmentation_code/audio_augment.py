import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Gain
import random
import torch
import torchaudio

seed = 42
random.seed(seed)
np.random.seed(seed)

# List of single augmentations
augmentations = [
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
    PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
    Gain(min_gain_db=-3, max_gain_db=3, p=1.0),
]

def augment_and_save_separate(input_path, output_dir, num_augments=2, sample_rate=16000):
    y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    chosen_augs = random.sample(augmentations, k=num_augments)
    for i, aug in enumerate(chosen_augs):
        # Apply the augmentation
        y_aug = aug(samples=y, sample_rate=sr)

        waveform = torch.from_numpy(y_aug)

        # Ensure it's 2D: [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)


        # Normalize to [-1.0, 1.0] if needed
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak

        out_fname = f"{base_name}_aug{i+1}.wav"
        out_path = os.path.join(output_dir, out_fname)
        # Save the waveform as 16-bit WAV
        torchaudio.save(out_path, waveform, sr, encoding="PCM_S", bits_per_sample=16)
        print(f"Saved: {out_path}")


# Paths
input_dir = "./raw_audio/train/"
output_dir = "./augmented_data/audio2"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".wav"):
        file_path = os.path.join(input_dir, fname)
        augment_and_save_separate(file_path, output_dir, num_augments=3)




