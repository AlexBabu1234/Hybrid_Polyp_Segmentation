import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

from config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, DEVICE, TRAIN_DATASETS
from dataset import PolypDataset
from models.attention_unet import AttentionUNet
from losses.hybrid_loss import HybridLoss
from utils.metrics import dice_score, iou_score


print("Device:", DEVICE)


# -----------------------------
# Load Datasets
# -----------------------------

train_sets = []

for name, paths in TRAIN_DATASETS.items():
    ds = PolypDataset(paths["images"], paths["masks"], augment=True)
    train_sets.append(ds)
    print(f"  {name}: {len(ds)} samples")

dataset = ConcatDataset(train_sets)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Train: {train_size} | Val: {val_size}")


# -----------------------------
# Model
# -----------------------------

model = AttentionUNet().to(DEVICE)

criterion = HybridLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    factor=0.5
)


# -----------------------------
# Storage for Graphs
# -----------------------------

train_losses = []
val_losses = []

dice_scores_list = []
iou_scores_list = []

y_scores = []
y_true = []

best_dice = 0


# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for imgs, masks in train_loader:

        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(imgs)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss = 0
    dice = 0
    iou = 0

    with torch.no_grad():

        for imgs, masks in val_loader:

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            dice += dice_score(outputs, masks).item()
            iou += iou_score(outputs, masks).item()

            probs = torch.sigmoid(outputs)

            y_scores.extend(probs.cpu().numpy().flatten())
            y_true.extend(masks.cpu().numpy().flatten())


    val_loss = val_loss / len(val_loader)
    dice = dice / len(val_loader)
    iou = iou / len(val_loader)

    val_losses.append(val_loss)
    dice_scores_list.append(dice)
    iou_scores_list.append(iou)

    scheduler.step(val_loss)


    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("Train Loss:", train_loss)
    print("Val Loss:", val_loss)
    print("Dice:", dice)
    print("IoU:", iou)


    # -----------------------------
    # Save Best Model
    # -----------------------------

    if dice > best_dice:

        best_dice = dice

        torch.save(model.state_dict(), "best_polyp_model.pth")

        print("Best model saved.")


# -----------------------------
# Save Results
# -----------------------------

y_scores = np.array(y_scores)
y_true = np.array(y_true)

np.save("y_scores.npy", y_scores)
np.save("y_true.npy", y_true)

epochs = range(1, EPOCHS + 1)


# Training vs Validation Loss
plt.figure()
plt.plot(epochs, train_losses)
plt.plot(epochs, val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.grid()
plt.savefig("loss_curve.png")
plt.close()


# Dice Score
plt.figure()
plt.plot(epochs, dice_scores_list)
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Validation Dice Score")
plt.grid()
plt.savefig("dice_curve.png")
plt.close()


# IoU Score
plt.figure()
plt.plot(epochs, iou_scores_list)
plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.title("Validation IoU Score")
plt.grid()
plt.savefig("iou_curve.png")
plt.close()


# Precision Recall Curve
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.grid()
plt.savefig("precision_recall_curve.png")
plt.close()


# ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.4f" % roc_auc)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve.png")
plt.close()


print("\n==============================")
print("FINAL RESULTS")
print("==============================")
print("Best Dice:", max(dice_scores_list))
print("Best IoU:", max(iou_scores_list))
print("ROC AUC:", roc_auc)
print("==============================")