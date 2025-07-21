import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ---------- Data Loading ----------
class FeatureDataset(Dataset):
    def __init__(self, feat_paths, label_path):
        # Load all modalities
        self.features = [pd.read_csv(p).values for p in feat_paths]
        self.labels = pd.read_csv(label_path).values.squeeze()

        # Standardize each modality
        self.scalers = [StandardScaler().fit(feat) for feat in self.features]
        self.features = [torch.tensor(scaler.transform(feat), dtype=torch.float32) 
                         for feat, scaler in zip(self.features, self.scalers)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [f[idx] for f in self.features], torch.tensor(self.labels[idx], dtype=torch.long)

# ---------- Self-Attention Module ----------
class SelfAttentionFusion(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x)
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)

# ---------- Cross-Modal Attention (MuLT-style) ----------
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, fused):
        attn_output, _ = self.attn(query, fused, fused)
        return self.norm(attn_output + query)

# ---------- Final Classifier ----------
class EDAMMModel(nn.Module):
    def __init__(self, input_dims, d_model=128, out_dim=2):
        super().__init__()
        self.input_proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in input_dims])
        self.self_attention = SelfAttentionFusion(d_model)
        self.cross_modal = nn.ModuleList([CrossModalAttention(d_model) for _ in input_dims])
        self.classifier = nn.Sequential(
            nn.Linear(len(input_dims) * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

    def forward(self, modalities):
        proj_modalities = [proj(x).unsqueeze(1) for x, proj in zip(modalities, self.input_proj)]
        combined = torch.cat(proj_modalities, dim=1)
        fused = self.self_attention(combined)

        enhanced = [att(x, fused).squeeze(1) for x, att in zip(proj_modalities, self.cross_modal)]
        y_plus = torch.cat(enhanced, dim=1)
        return self.classifier(y_plus)

# ---------- Training & Evaluation ----------
def train_eval(model, train_loader, test_loader, device='cpu', epochs=20, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = [x.to(device) for x in inputs]
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# ---------- Main ----------
if __name__ == "__main__":
    train_feat_paths = [
        "./encoders/encoeded/tabular/features.csv",
        "./encoders/encoeded/image/features.csv",
        # "train/word2vec_train.csv"
    ]
    train_labels = "train/labels_train.csv"

    test_feat_paths = [
        "test/wav2vec_test.csv",
        "test/tfidf_test.csv",
        "test/word2vec_test.csv"
    ]
    test_labels = "test/labels_test.csv"

    train_dataset = FeatureDataset(train_feat_paths, train_labels)
    test_dataset = FeatureDataset(test_feat_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    input_dims = [pd.read_csv(p).shape[1] for p in train_feat_paths]
    model = EDAMMModel(input_dims)

    train_eval(model, train_loader, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
