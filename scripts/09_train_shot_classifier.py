"""
================================================================================
FILE 09: train_shot_classifier.py
================================================================================

HOW TO RUN THIS FILE:
    python scripts/09_train_shot_classifier.py

WHAT THIS FILE DOES (super simple explanation):
    This file teaches a tiny robot brain (a small PyTorch neural network,
    called an "MLP") to guess what TYPE of shot happened -- smash, push,
    or not_shot -- just by looking at 3 numbers: how fast the wrist was
    moving, how long the wind-up took, and the elbow angle.

    You already hand-labeled ~33 moments (by watching the video and typing
    the answer into the "shot_type" column of the spreadsheet from file
    08). This file reads those labels, splits them into a "study" pile and
    a "quiz" pile, teaches the robot on the study pile, and tests it on
    the quiz pile it's never seen.

    IMPORTANT HONEST NOTE: 33 examples is a TINY amount for teaching a
    robot. It's enough to PROVE you know how to build and train a PyTorch
    model correctly, but not enough for the robot to become genuinely
    accurate on brand new videos. Expect (and it's totally fine to report)
    that it "memorizes" the study pile better and better, while doing WORSE
    on the quiz pile the longer it studies -- that's called overfitting,
    and seeing it happen here is actually a good sign you understand what's
    going on.

WHAT GOES IN, WHAT COMES OUT:
    IN:  data/shot_features_v2.csv    <- from file 08, WITH your hand-typed
                                          "shot_type" labels filled in
    OUT: outputs/shot_classifier.pt   <- the trained robot's "brain" (weights)

HOW THIS FILE CONNECTS TO OTHER FILES:
    08_calculate_contacts_v2.py
              |
              v
    data/shot_features_v2.csv  (+ your hand-typed shot_type labels)
              |
              v
    09_train_shot_classifier.py   <-- YOU ARE HERE
              |
              v
    outputs/shot_classifier.pt
              |
              v
    10_run_on_video.py   (loads this trained brain to label a NEW video)
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CSV_PATH = "data/shot_features_v2.csv"
FEATURE_COLS = ["wrist_speed", "windup_frames", "elbow_angle"]  # the 3 clues the robot gets to see
LABEL_COL = "shot_type"  # the answer we want the robot to learn to guess
RANDOM_SEED = 42          # fixed "shuffle seed" so results are repeatable, not random every run
EPOCHS = 200               # how many times the robot studies the whole study pile
BATCH_SIZE = 4             # dataset is tiny -- small batches on purpose
LEARNING_RATE = 0.01       # how big a "correction step" the robot takes each time it's wrong
HIDDEN_SIZE = 16           # size of the robot's "thinking layer" -- small model, small dataset

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------
class ShotDataset(Dataset):
    """
    SIMPLE EXPLANATION:
    PyTorch wants your data wrapped in a specific "Dataset" shape so it can
    feed it to the model piece by piece. This class just holds our numbers
    (X = the 3 clues) and answers (y = the shot type) and knows how to hand
    back one example at a time when asked.
    """
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
    """
    SIMPLE EXPLANATION:
    This IS the tiny robot brain. It takes in the 3 numbers (clues), passes
    them through one "thinking layer" of 16 little math units, and outputs
    a score for each possible answer (smash / push / not_shot). Whichever
    score is highest is the robot's guess.
    """
    def __init__(self, n_features, n_classes, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),  # clue numbers -> hidden thinking layer
            nn.ReLU(),                           # a simple "keep positive signals, zero out negative ones" step
            nn.Linear(hidden_size, n_classes),    # hidden layer -> one score per possible answer
        )

    def forward(self, x):
        return self.net(x)  # raw logits -- softmax happens inside CrossEntropyLoss


def load_data(csv_path):
    """
    SIMPLE EXPLANATION:
    Reads the spreadsheet, keeps ONLY the rows you actually hand-labeled
    (skips blank ones), cleans up the labels (so "Push" and "push" count as
    the SAME answer), and turns the text labels into numbers the robot can
    understand (LabelEncoder turns "smash"/"push"/"not_shot" into 0/1/2).
    """
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
    """
    SIMPLE EXPLANATION -- the overall recipe:
        1. Load your hand-labeled data
        2. Split it: some for STUDYING, some for a QUIZ (never studied)
        3. Scale the numbers so they're all on a similar range (helps
           the robot learn evenly, instead of one big number dominating)
        4. Study loop: show the robot the study pile 200 times, correcting
           it a little bit each time it's wrong
        5. Report how well it does on the study pile vs the quiz pile
        6. Save the trained brain to a file so file 10 can reuse it later
    """
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
    criterion = nn.CrossEntropyLoss()   # measures "how wrong" the robot's guess was
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # the "corrector"

    # ---------------- training loop ----------------
    # Each "epoch" = one full pass through the study pile.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()          # reset the correction from last time
            logits = model(xb)             # ask the robot to guess
            loss = criterion(logits, yb)   # measure how wrong the guess was
            loss.backward()                # figure out which direction to correct
            optimizer.step()               # actually apply the correction
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)

        # Every 20 epochs, check progress on BOTH piles (study and quiz)
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

    # Save the trained brain (the model's learned numbers) to a file, so we
    # don't have to retrain it every time -- file 10 will load this back up.
    torch.save(model.state_dict(), "outputs/shot_classifier.pt")
    print("Model weights saved to outputs/shot_classifier.pt")


def evaluate(model, loader):
    """
    SIMPLE EXPLANATION:
    Gives the robot a pile of examples WITHOUT telling it the answers,
    counts how many it got right, and returns that as a percentage
    (accuracy). Used for both the study pile and the quiz pile, so we can
    compare "does it just memorize?" vs "does it actually generalize?".
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():  # we're just checking, not learning right now
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


if __name__ == "__main__":
    main()