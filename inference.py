import torch
import cv2
import numpy as np

from config import DEVICE, MODEL_PATH, IMAGE_SIZE
from models.attention_unet import AttentionUNet


# -----------------------------
# Configuration
# -----------------------------

IMAGE_PATH = "test.jpg"  # replace with your test image


# -----------------------------
# Load Model
# -----------------------------

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Load & Preprocess Image
# -----------------------------

image = cv2.imread(IMAGE_PATH)
original = image.copy()

image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = image / 255.0
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
image = image.unsqueeze(0).to(DEVICE)


# -----------------------------
# Inference
# -----------------------------

with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output)
    mask = output.squeeze().cpu().numpy()

mask = (mask > 0.5).astype(np.uint8) * 255


# -----------------------------
# Resize & Overlay
# -----------------------------

mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

overlay = original.copy()
overlay[mask_resized == 255] = [0, 0, 255]

alpha = 0.5
output_image = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)


# -----------------------------
# Save Result
# -----------------------------

cv2.imwrite("output_mask.png", mask_resized)
cv2.imwrite("output_overlay.png", output_image)

print("Segmentation complete.")
print("Saved: output_mask.png")
print("Saved: output_overlay.png")