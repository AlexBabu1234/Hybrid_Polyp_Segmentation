import cv2
import torch
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

from config import DEVICE, MODEL_PATH, IMAGE_SIZE
from models.attention_unet import AttentionUNet
from gradcam import GradCAM


# -----------------------------
# Load Model
# -----------------------------

model = AttentionUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

gradcam = GradCAM(model)


# -----------------------------
# Preprocess Image
# -----------------------------

def preprocess(frame):
    image = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    return tensor


# -----------------------------
# Segmentation
# -----------------------------

def segment_frame(frame):
    input_tensor = preprocess(frame)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)
        mask = prob.squeeze().cpu().numpy()

    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]))

    overlay = frame.copy()
    overlay[mask_resized == 255] = [0, 0, 255]

    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Confidence = mean probability of detected polyp pixels only
    polyp_probs = mask[mask > 0.5]
    confidence = float(np.mean(polyp_probs)) if len(polyp_probs) > 0 else 0.0
    area = np.sum(mask_binary == 255) / mask_binary.size

    return result, mask_resized, confidence, area, input_tensor


# -----------------------------
# GradCAM
# -----------------------------

def generate_gradcam_overlay(frame, input_tensor):
    cam_heatmap = gradcam.generate(input_tensor)
    cam_resized = cv2.resize(cam_heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay


# -----------------------------
# GUI Setup
# -----------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Polyp Segmentation System")
app.geometry("1100x750")


# Display Panels
image_label = ctk.CTkLabel(app, text="")
image_label.pack(pady=20)

status_label = ctk.CTkLabel(app, text="Status: Idle")
status_label.pack()

confidence_label = ctk.CTkLabel(app, text="")
confidence_label.pack()

area_label = ctk.CTkLabel(app, text="")
area_label.pack()


def show_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((800, 500))
    imgtk = ImageTk.PhotoImage(img)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)


# Image Upload
def upload_image():
    path = filedialog.askopenfilename()
    if not path:
        return

    frame = cv2.imread(path)
    result, mask, conf, area, input_tensor = segment_frame(frame)

    show_image(result)

    status_label.configure(
        text="Polyp Detected" if conf > 50 else "No Polyp"
    )
    confidence_label.configure(text=f"Confidence: {conf:.1f}%")
    area_label.configure(text=f"Area: {area*100:.2f}%")


# Video Upload
video_capture = None
paused = False


def upload_video():
    global video_capture
    path = filedialog.askopenfilename()
    if not path:
        return
    video_capture = cv2.VideoCapture(path)
    process_video()


def process_video():
    global paused

    if paused:
        app.after(30, process_video)
        return

    ret, frame = video_capture.read()
    if not ret:
        return

    result, mask, conf, area, input_tensor = segment_frame(frame)
    show_image(result)

    status_label.configure(
        text="Polyp Detected" if conf > 50 else "No Polyp"
    )
    confidence_label.configure(text=f"Confidence: {conf:.1f}%")
    area_label.configure(text=f"Area: {area*100:.2f}%")

    app.after(30, process_video)


def toggle_pause():
    global paused
    paused = not paused


def save_result():
    file = filedialog.asksaveasfilename(defaultextension=".png")
    if file:
        img = image_label.imgtk
        image = ImageTk.getimage(img)
        image.save(file)


# Buttons
button_frame = ctk.CTkFrame(app)
button_frame.pack(pady=20)

ctk.CTkButton(button_frame, text="Upload Image", command=upload_image).grid(row=0, column=0, padx=10)
ctk.CTkButton(button_frame, text="Upload Video", command=upload_video).grid(row=0, column=1, padx=10)
ctk.CTkButton(button_frame, text="Pause/Resume", command=toggle_pause).grid(row=0, column=2, padx=10)
ctk.CTkButton(button_frame, text="Save Result", command=save_result).grid(row=0, column=3, padx=10)


app.mainloop()