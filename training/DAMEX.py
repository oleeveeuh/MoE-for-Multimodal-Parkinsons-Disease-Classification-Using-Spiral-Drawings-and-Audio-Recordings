import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from itertools import product
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

seed = 42  
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', nargs='+', required=True, help='List of dataset names')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = [
    'tabular',
    'image',
    'audio',
]
def evaluate_metrics(model, dataloader, device):
    preds_per_source = defaultdict(list)
    labels_per_source = defaultdict(list)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            for pred, label, src in zip(preds, y, source_ids):
                src = int(src.item())
                preds_per_source[src].append(pred.item())
                labels_per_source[src].append(label.item())

    # Plot confusion matrices per source
    # for src in sorted(preds_per_source.keys()):
    #     y_true = labels_per_source[src]
    #     y_pred = preds_per_source[src]

    #     cm = confusion_matrix(y_true, y_pred)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #     disp.plot(cmap=plt.cm.Blues)
    #     plt.title(f"Confusion Matrix for Source {src}")
    #     plt.show()

        # acc = accuracy_score(y_true, y_pred)
        # print(f"📊 Source {src} Accuracy: {acc:.4f}")

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    return acc, precision, recall, f1

# Get all predictions and true labels on test set
def get_preds_and_labels(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=1, gate_noise_std=1.0):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate_noise_std = gate_noise_std

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        if num_experts > 1:
            self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x, return_topk=False):
        # Case: Single expert — just apply it directly
        if self.num_experts == 1:
            output = self.experts[0](x)
            aux_loss = torch.tensor(0.0, device=x.device)
            topk = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            if return_topk:
                return output, aux_loss, topk
            else:
                return output, aux_loss

        # Case: MoE with routing
        logits = self.router(x)
        if self.training:
            noise = torch.randn_like(logits) * (self.gate_noise_std / self.num_experts)
            logits = logits + noise
        scores = F.softmax(logits, dim=-1)

        topk_vals, topk_inds = torch.topk(scores, self.k, dim=-1)
        dispatch_mask = torch.zeros_like(scores)
        dispatch_mask.scatter_(1, topk_inds, topk_vals)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        dispatch_mask = dispatch_mask.unsqueeze(-1)
        output = (expert_outputs * dispatch_mask).sum(dim=1)

        importance = dispatch_mask.sum(dim=0).squeeze()
        importance_mean = torch.mean(importance)
        importance_var = torch.var(importance)
        load_balancing_loss = importance_var / (importance_mean ** 2 + 1e-9)

        if return_topk:
            return output, load_balancing_loss, topk_inds
        else:
            return output, load_balancing_loss

class MoEClassifier(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k, num_classes):
        super().__init__()
        self.moe = MoELayer(input_dim, expert_dim, num_experts, k)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, return_topk=False):
        if return_topk:
            moe_out, aux_loss, topk = self.moe(x, return_topk=True)
            logits = self.classifier(moe_out)
            return logits, aux_loss, topk
        else:
            moe_out, aux_loss = self.moe(x)
            logits = self.classifier(moe_out)
            return logits, aux_loss

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        logits, aux_loss = model(x)
        loss = criterion(logits, y) + 0.01 * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    return total_loss / len(dataloader)

def evaluate_with_expert_tracking(model, dataloader, device):
    model.eval()
    expert_usage = defaultdict(lambda: torch.zeros(model.moe.num_experts))
    with torch.no_grad():
        for x, _, tag_batch in dataloader:
            x = x.to(device)
            _, _, topk = model(x, return_topk=True)
            for idx, tag in zip(topk.squeeze().tolist(), tag_batch.tolist()):
                expert_usage[tag][idx] += 1
    return expert_usage

def plot_expert_heatmap(expert_usage_dict):
    df = pd.DataFrame(expert_usage_dict).T
    df.columns = [f"Expert {i}" for i in range(df.shape[1])]
    df_norm = df.div(df.sum(axis=1), axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_norm, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Expert Usage per Dataset (Normalized)")
    plt.xlabel("Experts")
    plt.ylabel("Datasets")
    plt.tight_layout()
    plt.show()

def load_csv_features_and_labels(feature_path, label_path):
    feature_arrays = pd.read_csv(feature_path).values.astype(np.float32)
    X = (feature_arrays)
    y = pd.read_csv(label_path).values.squeeze()
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)

all_features = []
all_labels = []
test_features = []
test_labels = []
all_sources = []  # optional: for SourceConcatDataset or expert analysis
test_sources = []  # optional: for SourceConcatDataset or expert analysis
all_results = []
source_id = 0
criterion = nn.CrossEntropyLoss()

   
for d in datasets:
    feat_path = f"./encoders/encoded/best/{d}/train_features.csv"
    label_path = f"./encoders/encoded/best/{d}/train_labels.csv"
    test_feat_path = f"./encoders/encoded/best/{d}/test_features.csv"
    test_label_path = f"./encoders/encoded/best/{d}/test_labels.csv"

    X, y = load_csv_features_and_labels(feat_path, label_path)
    all_features.append(X)
    all_labels.append(y)
    all_sources.extend([source_id] * len(y))

    test_x, test_y = load_csv_features_and_labels(test_feat_path, test_label_path)
    test_features.append(test_x)
    test_labels.append(test_y)
    test_sources.extend([source_id] * len(test_y))

    source_id += 1

X_all = torch.cat(all_features, dim=0)
y_all = torch.cat(all_labels, dim=0)
X_test = torch.cat(test_features, dim=0)
y_test = torch.cat(test_labels, dim=0)
source_ids = torch.tensor(all_sources)
test_source_ids = torch.tensor(test_sources)
num_experts_list = [1]
k_list = [1]
patience = 5

# Create dataset with source tracking
full_dataset = TensorDataset(X_all, y_all, source_ids)
test_dataset = TensorDataset(X_test, y_test, test_source_ids)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# plot training loss


for num_experts, k in product(num_experts_list, k_list):
    print(f"\n🧪 Testing num_experts = {num_experts}, k = {k}")
    all_results = []
    best_model = None
    best_acc = 0


    if k > num_experts:
        continue

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all)):
        patience_counter = 0
        print(f"\nFold {fold + 1}/{5}")

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        input_dim = X_all.shape[1]
        num_classes = len(torch.unique(y_all))

        model = MoEClassifier(
            input_dim=input_dim,
            expert_dim=192,        # or tune this
            num_experts=num_experts,        # or your desired number
            k=k,                  # typically 1 or 2
            num_classes=num_classes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(200):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            # print(f"Fold {fold+1}, Epoch {epoch+1}: Loss = {loss:.4f}")

        # Evaluate metrics
        print(f"📊 Metrics for Fold {fold+1}:")
        acc, precision, recall, f1 = evaluate_metrics(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        all_results.append([acc, precision, recall, f1])
        # expert_usage = evaluate_with_expert_tracking(model, test_loader, device)
        # plot_expert_heatmap(expert_usage)

    results_array = np.array(all_results)  # shape: [num_folds, 4]
    mean_metrics = results_array.mean(axis=0)  # mean across folds

    print(f"Accuracy:{mean_metrics[0]:.4f}, Precision:{mean_metrics[1]:.4f}, Recall:{mean_metrics[2]:.4f}, F1:{mean_metrics[3]:.4f}")

    model.load_state_dict(torch.load('best_model.pt'))

    print("📊 Evaluating on TEST set:")
    test_acc, test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader, device)

    # Get preds and true labels
    test_preds, test_labels = get_preds_and_labels(model, test_loader, device)

    # Compute confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.show()
