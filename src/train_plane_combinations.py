import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from dataset import MRNetACLDataset, collate_fn
from model import MRNetACLModel


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\MajorProject_B12\data\dataset1"

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4

SEED = 42

RESULTS_DIR = r"C:\MajorProject_B12\results"
CHECKPOINT_DIR = r"C:\MajorProject_B12\checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# ALL 7 PLANE COMBINATIONS
# ============================================================

PLANE_COMBINATIONS = [
    ["axial"],
    ["coronal"],
    ["sagittal"],
    ["axial", "coronal"],
    ["axial", "sagittal"],
    ["coronal", "sagittal"],
    ["axial", "coronal", "sagittal"],
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MRNet ACL - PLANE COMBINATION EXPERIMENT")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Seed:", SEED)

print()
print("Experiments to run:", len(PLANE_COMBINATIONS))

for i, planes in enumerate(PLANE_COMBINATIONS, 1):
    print(f"{i}. {' + '.join(planes)}")

print("=" * 70)


# ============================================================
# FUNCTION: TRAIN ONE PLANE COMBINATION
# ============================================================

def run_experiment(planes, experiment_number):

    print()
    print()
    print("#" * 70)
    print(
        f"EXPERIMENT {experiment_number}/7: "
        f"{' + '.join(planes).upper()}"
    )
    print("#" * 70)

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = MRNetACLDataset(
        DATASET_PATH,
        split="train",
        planes=tuple(planes)
    )

    valid_dataset = MRNetACLDataset(
        DATASET_PATH,
        split="valid",
        planes=tuple(planes)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MRNetACLModel(
        planes=planes
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    labels = train_dataset.labels["acl"].values

    num_negative = np.sum(labels == 0)
    num_positive = np.sum(labels == 1)

    pos_weight = num_negative / num_positive

    print("Negative samples:", num_negative)
    print("Positive samples:", num_positive)
    print("Positive class weight:", pos_weight)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device
        )
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_accuracy = 0.0
    best_epoch = 0

    plane_name = "_".join(planes)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"best_{plane_name}.pth"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_idx, (volumes, labels) in enumerate(
            train_loader
        ):

            for plane in volumes:
                volumes[plane] = volumes[plane].to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(volumes)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()

        average_loss = (
            running_loss / len(train_loader)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():

            for volumes, labels in valid_loader:

                for plane in volumes:
                    volumes[plane] = volumes[plane].to(device)

                labels = labels.to(device)

                outputs = model(volumes)

                probabilities = torch.sigmoid(outputs)

                predictions = (
                    probabilities >= 0.5
                ).float()

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

        epoch_accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.4f} "
            f"| Validation Accuracy: "
            f"{epoch_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE BEST CHECKPOINT
        # ----------------------------------------------------

        if epoch_accuracy > best_accuracy:

            best_accuracy = epoch_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "planes": planes,
                    "validation_accuracy": best_accuracy,
                    "seed": SEED,
                },
                checkpoint_path
            )

            print(
                f"*** Best model saved: "
                f"{best_accuracy * 100:.2f}% ***"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for volumes, labels in valid_loader:

            for plane in volumes:
                volumes[plane] = volumes[plane].to(device)

            labels = labels.to(device)

            outputs = model(volumes)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).float()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        all_labels,
        all_predictions
    ).ravel()

    specificity = tn / (tn + fp)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print("EXPERIMENT RESULTS")
    print("-" * 70)

    print("Plane combination:", " + ".join(planes))
    print("Best epoch:", best_epoch)

    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Precision:   {precision * 100:.2f}%")
    print(f"Sensitivity: {sensitivity * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"F1 Score:    {f1 * 100:.2f}%")
    print(f"AUROC:       {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(f"TN: {tn}   FP: {fp}")
    print(f"FN: {fn}   TP: {tp}")

    print("Parameters:", total_params)

    print("-" * 70)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "planes": " + ".join(planes),
        "num_planes": len(planes),
        "parameters": total_params,
        "best_epoch": best_epoch,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auroc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# RUN ALL 7 EXPERIMENTS
# ============================================================

results = []

for experiment_number, planes in enumerate(
    PLANE_COMBINATIONS,
    start=1
):

    result = run_experiment(
        planes,
        experiment_number
    )

    results.append(result)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_file = os.path.join(
    RESULTS_DIR,
    "plane_combination_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print()
print("=" * 100)
print("FINAL PLANE COMBINATION COMPARISON")
print("=" * 100)

print(
    results_df[
        [
            "planes",
            "parameters",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "auroc",
        ]
    ].to_string(index=False)
)

print()
print("Results saved to:")
print(results_file)

print()
print("=" * 100)
print("ALL 7 EXPERIMENTS COMPLETE")
print("=" * 100)