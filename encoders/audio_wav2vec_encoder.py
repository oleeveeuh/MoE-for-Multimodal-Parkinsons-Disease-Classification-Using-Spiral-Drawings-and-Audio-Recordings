import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import pandas as pd
import os
import csv

# Initialize model & processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Linear projection layer from 768 → 192 features
projector = torch.nn.Linear(768, 192)
projector.eval()

def extract_projected_features(filepath):
    # Load waveform (mono)
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    input_values = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_values  # batch size 1

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state  # shape: (1, time_steps, 768)

        # Mean pool across time dimension
        mean_pooled = hidden_states.mean(dim=1)  # shape: (1, 768)

        # Linear projection 768 → 192
        projected = projector(mean_pooled)  # shape: (1, 192)

        # Convert to 1D numpy array
        features = projected.squeeze().cpu().numpy()

    return features

def save_features_to_csv(features, csv_file):
    # Append features as a new row in the CSV file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features)

# Example usage
audio_folder = "./test"
csv_path = "./featurestest.csv"
all_labels = []
# Make sure CSV file exists (write header optional)
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"feat_{i+1}" for i in range(192)])  # optional header

# Process all wav files and save features
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_folder, filename)
        feats = extract_projected_features(filepath)
        save_features_to_csv(feats, csv_path)
        if filename.startswith("P"):
          all_labels.append(1)
        else:
          all_labels.append(0)
        print(f"Processed {filename}")


all_labels = torch.cat(all_labels, dim=0)      # [N]
labels_df = pd.DataFrame(all_labels.cpu().numpy())

labels_df.to_csv('./labelstest.csv', index=False)