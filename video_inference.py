import torch
import cv2
import numpy as np

from config import DEVICE, MODEL_PATH, IMAGE_SIZE
from models.attention_unet import AttentionUNet


# -----------------------------
# Configuration
# -----------------------------

VIDEO_PATH = "test_video.mp4"  # replace with your video


# -----------------------------
# Load Model
# -----------------------------

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Process Video
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # Preprocess
    image = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = image.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        mask = output.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

    overlay = original.copy()
    overlay[mask_resized == 255] = [0, 0, 255]

    alpha = 0.4
    output_frame = cv2.addWeighted(original, 1 - alpha, overlay, alpha, 0)

    cv2.imshow("Polyp Segmentation - Real Time", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()