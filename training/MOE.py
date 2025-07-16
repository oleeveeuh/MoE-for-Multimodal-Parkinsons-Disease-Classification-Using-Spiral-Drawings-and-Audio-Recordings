import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = 'cpu'
# ---- Dataset ----
class MultiModalDataset(Dataset):
    def __init__(self, feature_paths, label_path):
        # feature_paths: dict modality -> csv path
        self.features = {}
        for m, p in feature_paths.items():
            self.features[m] = pd.read_csv(p).values.astype(np.float32)
        self.labels = pd.read_csv(label_path).values.squeeze()
        assert all(len(v) == len(self.labels) for v in self.features.values())
        self.modalities = list(self.features.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return dict modality->feature tensor + label tensor
        feats = {m: torch.tensor(self.features[m][idx]) for m in self.modalities}
        label = torch.tensor(self.labels[idx]).long()
        return feats, label

# ---- Model ----
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    def forward(self, x):
        weights = F.softmax(self.fc(x), dim=1)
        return weights

class MixtureOfExperts(nn.Module):
    def __init__(self, expert_input_dims, hidden_dim=128, expert_output_dim=64, num_classes=2):
        super().__init__()
        self.num_experts = len(expert_input_dims)
        self.experts = nn.ModuleList([Expert(in_dim, hidden_dim, expert_output_dim) for in_dim in expert_input_dims])
        self.gating = GatingNetwork(sum(expert_input_dims), self.num_experts)
        self.classifier = nn.Linear(expert_output_dim, num_classes)

    def forward(self, inputs):
        # inputs: list of tensors [B, dim_i]
        expert_outputs = [expert(x) for expert, x in zip(self.experts, inputs)]  # list of [B, output_dim]
        gating_input = torch.cat(inputs, dim=1)  # [B, sum input dims]
        weights = self.gating(gating_input)      # [B, num_experts]
        stacked = torch.stack(expert_outputs, dim=1)  # [B, num_experts, output_dim]
        weighted_sum = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [B, output_dim]
        out = self.classifier(weighted_sum)  # [B, num_classes]
        return out, weights

# ---- Training & Evaluation ----
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for feats, labels in tqdm(dataloader):
        inputs = [feats[m].to(device) for m in feats]
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader.dataset), acc

def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in dataloader:
            inputs = [feats[m].to(device) for m in feats]
            labels = labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(dataloader.dataset), acc, prec, rec, f1

# ---- Main ----
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = './data'  # Adjust as needed
    modalities = ['wav2vec', 'tfidf', 'word2vec']  # Change if needed (e.g. for ablation)
    train_paths = {m: os.path.join(data_root, 'train', f'{m}.csv') for m in modalities}
    test_paths = {m: os.path.join(data_root, 'test', f'{m}.csv') for m in modalities}
    train_label_path = os.path.join(data_root, 'train', 'labels.csv')
    test_label_path = os.path.join(data_root, 'test', 'labels.csv')

    # Load datasets
    train_dataset = MultiModalDataset(train_paths, train_label_path)
    test_dataset = MultiModalDataset(test_paths, test_label_path)

    # Split validation from train (e.g. 10%)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Determine input dims per modality
    expert_input_dims = []
    for m in modalities:
        expert_input_dims.append(train_dataset.dataset.features[m].shape[1])

    model = MixtureOfExperts(expert_input_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f} prec: {val_prec:.4f} rec: {val_rec:.4f} f1: {val_f1:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_moe_model.pth')

    # Load best model and test
    model.load_state_dict(torch.load('best_moe_model.pth'))
    test_loss, test_acc, test_prec, test_rec, test_f1 = eval_model(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} acc: {test_acc:.4f} prec: {test_prec:.4f} rec: {test_rec:.4f} f1: {test_f1:.4f}")

if __name__ == '__main__':
    main()
