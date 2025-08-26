import torch
from transformers import Wav2Vec2Model
import torchaudio
import torch.nn as nn
from transformers import AutoProcessor, HubertModel, WavLMModel
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, HubertModel  # or WavLMModel

import os
import csv

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class FeatureReducer(nn.Module):
    def __init__(self, input_dim=768, output_dim=192, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):  # x: [seq_len, hidden_dim]
        if self.pooling == 'mean':
            x = x.mean(dim=0)         # → [hidden_dim]
        elif self.pooling == 'max':
            x = x.max(dim=0).values   # → [hidden_dim]
        else:
            raise ValueError("Invalid pooling type")

        x = self.proj(x)              # → [output_dim]
        return x.unsqueeze(0)         # → [1, output_dim]


def extract_features(audio_tensor, feature_extractor, model):
    inputs = feature_extractor(audio_tensor, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: (batch_size=1, sequence_length, hidden_dim)
    return outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_size]

def load_audio(path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # [T]

def extract_projected_features(path, model_type='hubert'):
    audio = load_audio(path)

    features = extract_features(audio, feature_extractor, model)

    flattened = reducer(features)

    return flattened  # [seq_len, hidden_size]


def save_features_to_csv(features, csv_file):
    # Append features as a new row in the CSV file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features.squeeze(0).tolist())

# Example usage
audio_folder = "./augmented_data/audio2"
# train_folder = "./raw_audio/train"
# test_folder = "raw_audio/test"

# audio_folder = "./augmented_data/preaug_audio"
train_folder = "./preprocessed_data/audio/train"
test_folder = "preprocessed_data/audio/test"


files_dir1 = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if os.path.isfile(os.path.join(audio_folder, f))]
files_dir2 = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]
all_files = files_dir1 + files_dir2

test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
reducer = FeatureReducer(input_dim=768, output_dim=192, pooling='mean')
reducer.eval()

model_names = {
    "wav2vec2",
    "hubert",
    "wavlm",
}

for name in model_names:
    print(f"\n🔁 Training model: {name}")
    model = None
    feature_extractor = None
    csv_path = (f"./encoders/encoded/audio/{name}_train_features2.csv")
    test_csv_path = csv_path.replace("train", "test")
    train_labels = []
    test_labels = []

    # Load model and processor
    if name == "hubert":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    elif name == "wav2vec2":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    elif name == "wavlm":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
        model = WavLMModel.from_pretrained("microsoft/wavlm-base")

    model.eval()



    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"feat_{i+1}" for i in range(192)])  # optional header
    else:
        open(csv_path, 'w').close()
        open(test_csv_path, 'w').close()
        print("Cleared")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"feat_{i+1}" for i in range(192)])  # optional header

        with open(test_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"feat_{i+1}" for i in range(192)])  # optional header


    # Process all wav files and save features
    for filename in all_files:
        if filename.endswith(".wav"):
            feats = extract_projected_features(filename)
            save_features_to_csv(feats, csv_path)
            if "PD_" in filename:
                train_labels.append(1)
            else:
                train_labels.append(0)
            print(f"Processed {filename}")

    for filename in test_files:
        if filename.endswith(".wav"):
            feats = extract_projected_features(filename)
            save_features_to_csv(feats, test_csv_path)
            if "PD_" in filename:
                test_labels.append(1)
            else:
                test_labels.append(0)
            print(f"Processed {filename}")

    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    train_labels_df = pd.DataFrame(train_labels.cpu().numpy())
    test_labels_df = pd.DataFrame(test_labels.cpu().numpy())

    train_labels_df.to_csv(csv_path.replace("features", "labels"), index=False)
    test_labels_df.to_csv(test_csv_path.replace("features", "labels"), index=False)

