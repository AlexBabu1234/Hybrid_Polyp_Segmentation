import os
import io
import base64
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
import tempfile

from config import DEVICE, MODEL_PATH, IMAGE_SIZE
from models.attention_unet import AttentionUNet
from gradcam import generate_gradcam

# -----------------------------
# Flask App
# -----------------------------

app = Flask(
    __name__,
    static_folder="static",
    template_folder="static"
)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB for video uploads

# -----------------------------
# Load Model Once
# -----------------------------

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print(f"Model loaded on {DEVICE}")


def decode_image(data_url):
    """Decode base64 data URL to OpenCV BGR image."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image(img):
    """Encode OpenCV BGR image to base64 data URL."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_inference(frame):
    """Run segmentation on a BGR frame. Returns results dict."""
    original = frame.copy()

    # Preprocess
    image = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # Segmentation
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = torch.sigmoid(output)
        mask_prob = prob_map.squeeze().cpu().numpy()

    # Confidence = mean probability of detected polyp pixels (not full image)
    polyp_probs = mask_prob[mask_prob > 0.5]
    if len(polyp_probs) > 0:
        confidence = float(np.mean(polyp_probs)) * 100
    else:
        confidence = 0.0
    binary_mask = (mask_prob > 0.5).astype(np.uint8)
    area_percentage = (np.sum(binary_mask) / binary_mask.size) * 100

    # Overlay
    mask_resized = cv2.resize(binary_mask, (original.shape[1], original.shape[0]))
    overlay = original.copy()
    overlay[mask_resized == 1] = [0, 0, 255]
    seg_result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    # Mask as white-on-black
    mask_display = (mask_resized * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_INFERNO)

    # GradCAM
    image_tensor_grad = torch.tensor(
        cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    image_tensor_grad.requires_grad = True
    gradcam_overlay = generate_gradcam(model, image_tensor_grad, original)

    status = "Polyp Detected" if area_percentage > 1.0 else "No Polyp Detected"

    return {
        "segmentation": encode_image(seg_result),
        "mask": encode_image(mask_colored),
        "gradcam": encode_image(gradcam_overlay),
        "confidence": round(confidence, 2),
        "area": round(area_percentage, 2),
        "status": status,
    }


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def segment():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        frame = decode_image(image_data)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        results = run_inference(frame)

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model": "AttentionUNet (Hybrid)"
    })


@app.route("/api/segment-frame", methods=["POST"])
def segment_frame():
    """Lightweight endpoint for video frames — skips GradCAM for speed."""
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        frame = decode_image(image_data)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        original = frame.copy()

        # Preprocess
        image = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        image = image / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Segmentation
        with torch.no_grad():
            output = model(image_tensor)
            prob_map = torch.sigmoid(output)
            mask_prob = prob_map.squeeze().cpu().numpy()

        polyp_probs = mask_prob[mask_prob > 0.5]
        confidence = float(np.mean(polyp_probs)) * 100 if len(polyp_probs) > 0 else 0.0
        binary_mask = (mask_prob > 0.5).astype(np.uint8)
        area_percentage = (np.sum(binary_mask) / binary_mask.size) * 100

        # Overlay
        mask_resized = cv2.resize(binary_mask, (original.shape[1], original.shape[0]))
        overlay = original.copy()
        overlay[mask_resized == 1] = [0, 0, 255]
        seg_result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

        status = "Polyp Detected" if area_percentage > 1.0 else "No Polyp Detected"

        return jsonify({
            "segmentation": encode_image(seg_result),
            "confidence": round(confidence, 2),
            "area": round(area_percentage, 2),
            "status": status,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
