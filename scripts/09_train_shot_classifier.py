"""
Proof-of-concept shot-type classifier (PyTorch MLP).

GOAL: demonstrate understanding of PyTorch basics (Dataset/DataLoader,
nn.Module, training loop) -- NOT to build a production-accurate classifier.

Dataset: ~30 hand-labeled candidates from data/shot_features_v2.csv, labeled
by watching outputs/pose_output.mp4 at each `approx_second`. This is far too
small and imbalanced to expect a model that generalizes -- that's expected
and worth saying directly in an interview, not something to hide.

Features used: wrist_speed, windup_frames, elbow_angle
  (numeric, directly available in the CSV -- no extra feature engineering
  needed for a first pass)

Labels used: shot_type column (e.g. not_shot / push / smash)
  Rows with an empty shot_type are dropped -- only hand-labeled rows count.

Run: python scripts/09_train_shot_classifier.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CSV_PATH = "data/shot_features_v2.csv"
FEATURE_COLS = ["wrist_speed", "windup_frames", "elbow_angle"]
LABEL_COL = "shot_type"
RANDOM_SEED = 42
EPOCHS = 200
BATCH_SIZE = 4          # dataset is tiny -- small batches on purpose
LEARNING_RATE = 0.01
HIDDEN_SIZE = 16         # small model, small dataset -- avoid overfitting further than necessary

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------
class ShotDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------------
# Model -- a small MLP: input -> hidden (ReLU) -> output logits
# ------------------------------------------------------------------------
class ShotClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits -- softmax happens inside CrossEntropyLoss


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # normalize any stray capitalization ("Push" vs "push") before dropping
    # empties, so labels aren't accidentally split into extra classes
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()
    df = df[df[LABEL_COL].notna() & (df[LABEL_COL] != "") & (df[LABEL_COL] != "nan")]

    df = df.dropna(subset=FEATURE_COLS)  # audio_only rows have no pose features

    print(f"Loaded {len(df)} labeled rows")
    print(df[LABEL_COL].value_counts().to_string())

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def main():
    X, y, encoder = load_data(CSV_PATH)
    n_classes = len(encoder.classes_)
    print(f"Classes: {list(encoder.classes_)}")

    # Small dataset -> a plain train/test split, stratified so every class
    # shows up in both halves where possible. With ~30 rows this split is
    # for demonstration, not a reliable accuracy estimate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # Standardize features using ONLY training stats, then apply to test --
    # avoids leaking test-set information into the scaler.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = ShotDataset(X_train, y_train)
    test_ds = ShotDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ShotClassifier(n_features=len(FEATURE_COLS), n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------- training loop ----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)

        if epoch % 20 == 0 or epoch == 1:
            train_acc = evaluate(model, train_loader)
            test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch:3d} | train_loss={avg_loss:.4f} "
                  f"| train_acc={train_acc:.2f} | test_acc={test_acc:.2f}")

    # ---------------- final report ----------------
    print("\nFinal evaluation:")
    final_test_acc = evaluate(model, test_loader)
    print(f"Test accuracy: {final_test_acc:.2f} "
          f"(on {len(test_ds)} held-out samples -- too small to be a "
          f"reliable estimate, reported here for completeness only)")

    torch.save(model.state_dict(), "outputs/shot_classifier.pt")
    print("Model weights saved to outputs/shot_classifier.pt")


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


if __name__ == "__main__":
    main()