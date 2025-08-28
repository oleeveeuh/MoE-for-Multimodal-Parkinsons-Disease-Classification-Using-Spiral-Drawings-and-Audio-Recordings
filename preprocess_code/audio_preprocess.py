import torchaudio
import torchaudio.transforms as T
import torch
import os
from sklearn.model_selection import train_test_split
import shutil

# torchaudio.set_audio_backend("soundfile")  # or "soundfile"
import librosa

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

def preprocess_and_save_wav2vec(filepath, output_path, target_sr=16000):
    # Load waveform and original sample rate
    # waveform, sr = torchaudio.load(filepath)  # shape: (channels, samples)
    waveform, sr = librosa.load(filepath, sr=None)
    waveform = torch.from_numpy(waveform)

    # Ensure it's 2D: [channels, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr


    # Normalize to [-1.0, 1.0] if needed
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak

    # Save the waveform as 16-bit WAV
    torchaudio.save(output_path, waveform, sr, encoding="PCM_S", bits_per_sample=16)

    print(f"Saved preprocessed file to: {output_path}")


processed_images = []
labels = []
healthy_path = "./raw_audio/HC_AH"
pd_path = "./raw_audio/PD_AH"
save_path = './raw_audio'
os.makedirs(save_path, exist_ok=True)


for file in os.listdir(healthy_path):
    file_path = os.path.join(healthy_path, file)

    new_save_path = os.path.join(save_path, ('H_' + file))
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    if file.lower().endswith(('.wav')):
        # processed_images.append(new_save_path)
        processed_images.append(file)

        labels.append(0)
        # preprocess_and_save_wav2vec(file_path, new_save_path)

for file in os.listdir(pd_path):
    file_path = os.path.join(pd_path, file)
    new_save_path = os.path.join(save_path, ('PD_' + file))
    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
      
    if file.lower().endswith(('.wav')):
        # processed_images.append(new_save_path)
        processed_images.append(file)

        labels.append(1)
        # preprocess_and_save_wav2vec(file_path, new_save_path)


X_train, X_test, y_train, y_test = train_test_split(processed_images, labels, test_size=0.2, random_state=42, stratify=labels)

clear_directory("./raw_audio/train")
clear_directory("./raw_audio/test")

for path in X_train:
    new_save_path = "./raw_audio/train"

    if path in os.listdir(pd_path):
        new_save_path = os.path.join(new_save_path, f'PD_{path}')
        path = os.path.join(pd_path, path)
    else: 
        new_save_path = os.path.join(new_save_path, f'H_{path}')
        path = os.path.join(healthy_path, path)


    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    shutil.copy(path,new_save_path)


for path in X_test:
    new_save_path = "./raw_audio/test"

    if path in os.listdir(pd_path):
        new_save_path = os.path.join(new_save_path, f'PD_{path}')
        origin_path = os.path.join(pd_path, path)

    else: 
        new_save_path = os.path.join(new_save_path, f'H_{path}')
        origin_path = os.path.join(healthy_path, path)

    os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
    
    shutil.copy(origin_path,new_save_path)