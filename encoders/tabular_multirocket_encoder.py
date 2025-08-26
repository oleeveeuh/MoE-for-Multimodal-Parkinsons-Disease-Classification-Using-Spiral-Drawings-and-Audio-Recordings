import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import MultiRocketMultivariate
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

def dataset_to_numpy(dataset):
    X, y = [], []
    for i in range(len(dataset)):
        data, label = dataset[i]  # data: [T, F]
        data = data.T.numpy()     # Convert to [F, T]
        X.append(data)
        y.append(label)
    return np.stack(X), np.array(y)

class MeanPaddedTimeSeriesDataset(Dataset):
    def __init__(self, folder_path, label_map=None):
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
        self.label_map = label_map or self._build_label_map()
        self.mean_len = self._compute_mean_length()

    def _build_label_map(self):
        labels = set(os.path.basename(f).split('_')[0] for f in self.file_paths)
        return {label: idx for idx, label in enumerate(sorted(labels))}

    def _compute_mean_length(self):
        lengths = []
        for file in self.file_paths:
            with open(file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                lengths.append(len(lines))
        return int(round(np.mean(lengths)))

    def _load_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        data = [list(map(float, line.split(';'))) for line in lines]
        return np.array(data)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        raw_data = self._load_file(path)
        T, F = raw_data.shape
        target_len = self.mean_len

        # Truncate or pad
        if T > target_len:
            data = raw_data[:target_len]
        elif T < target_len:
            pad_amount = target_len - T
            pad = np.zeros((pad_amount, F))
            data = np.concatenate([raw_data, pad], axis=0)
        else:
            data = raw_data

        # Get label from filename
        label_name = os.path.basename(path).split('_')[0]
        label = self.label_map[label_name]

        return torch.tensor(data, dtype=torch.float32), label

    def __len__(self):
        return len(self.file_paths)


dataset = MeanPaddedTimeSeriesDataset("path/to/data")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Load and split dataset
dataset = MeanPaddedTimeSeriesDataset("path/to/your/data")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Convert to numpy
X_train, y_train = dataset_to_numpy(train_set)
X_val, y_val = dataset_to_numpy(val_set)

# Fit transformer
transformer = MultiRocketMultivariate(num_kernels=10000)
transformer.fit(X_train)
X_train_feat = transformer.transform(X_train)
X_val_feat = transformer.transform(X_val)

# Early stopping logic
best_acc = 0.0
best_model = None
patience = 3
counter = 0

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)

rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,                  # number of different combinations to try
    cv=5,                       # 5-fold cross-validation
    scoring='accuracy',        # or f1, roc_auc depending on your task
    verbose=1,
    n_jobs=-1,                 # use all CPUs
    random_state=42
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

for n_estimators in n_estimators_list:
    print(f"Training Random Forest with {n_estimators} trees...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train_feat, y_train)
    val_pred = clf.predict(X_val_feat)
    acc = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = clf
        dump(clf, "best_model.joblib")
        dump(transformer, "rocket_transformer.joblib")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print(f"Best Validation Accuracy: {best_acc:.4f}")


# Reload saved model and transformer
best_model = load("best_model.joblib")
transformer = load("rocket_transformer.joblib")

# Convert test data
X_test, y_test = dataset_to_numpy(val_set)  # or a new unseen test set
X_test_feat = transformer.transform(X_test)
y_pred = best_model.predict(X_test_feat)

acc = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {acc:.4f}")
