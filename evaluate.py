import torch
from torch.utils.data import DataLoader

from config import DEVICE, MODEL_PATH, EVAL_DATASETS
from dataset import PolypDataset
from models.attention_unet import AttentionUNet
from utils.metrics import dice_score, iou_score


# -----------------------------
# Load Model
# -----------------------------

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("Model loaded from:", MODEL_PATH)


# -----------------------------
# Evaluate
# -----------------------------

def evaluate_dataset(name, img_path, mask_path):

    dataset = PolypDataset(img_path, mask_path, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_total = 0
    iou_total = 0

    with torch.no_grad():

        for imgs, masks in loader:

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)

            dice_total += dice_score(outputs, masks).item()
            iou_total += iou_score(outputs, masks).item()

    dice = dice_total / len(loader)
    iou = iou_total / len(loader)

    print(f"\n{name}")
    print(f"  Dice: {dice:.4f}")
    print(f"  IoU : {iou:.4f}")
    print("  -----------------------")

    return dice, iou


# -----------------------------
# Run All Evaluations
# -----------------------------

print("\n========== Cross-Dataset Evaluation ==========\n")

for name, paths in EVAL_DATASETS.items():
    evaluate_dataset(name, paths["images"], paths["masks"])