# Hybrid Polyp Segmentation

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

An advanced, end-to-end web application for the real-time detection, segmentation, and analysis of colonoscopy polyps. Designed for medical professionals and researchers, this tool utilizes deep learning to provide precise bounding boxes, segmentation masks, and diagnostic confidence scores.

## ✨ Key Features

- **Real-Time Video & Image Processing**: Instantly segment polyps from uploaded images or clinical colonoscopy videos.
- **Grad-CAM Visualization**: Transparent AI decision making with heatmap attribution overlays to see exactly *where* the model is looking.
- **Robust Scoring System**: Evaluates positive confidence using specialized algorithms that average probabilities exclusively across the detected polyp area.
- **Automated Diagnostic Reports**: Instantly generates downloadable PDF medical reports containing the predicted masks, heatmaps, and diagnostic metrics.
- **Modern Web Interface**: A beautifully designed, fully responsive web dashboard with drag-and-drop uploads and instant inference.

## 🛠️ Technologies Used

- **Deep Learning Architecture**: Attention U-Net
- **Backend Framework**: Flask (Python)
- **Computer Vision**: OpenCV, PyTorch, TorchVision
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla with Tailwind elements)
- **Visualization**: Matplotlib (for Grad-CAM and Report Generation)

## 📦 Setup & Installation

**Prerequisites**: Make sure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlexBabu1234/Hybrid_Polyp_Segmentation.git
   cd Hybrid_Polyp_Segmentation
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may need to manually install the PyTorch version that matches your CUDA drivers for GPU acceleration).*

4. **Add the Model Weights:**
   Place your trained model weight file (`best_unet_model.pth`) directly into the project root directory.

## 🚀 Usage

1. **Start the Web Server:**
   ```bash
   python web_app.py
   ```
2. Open your web browser and navigate to: `http://127.0.0.1:5000`
3. Drag and drop a colonoscopy image (JPG/PNG) or video (MP4) to begin the automated segmentation process!

## 🧠 Model Architecture

This project is built around an **Attention U-Net** structure.
Standard U-Nets are excellent for biomedical image segmentation, but the *Attention* mechanism allows the network to suppress irrelevant background regions and strongly focus on target structures (in this case, polyps) of varying shapes and sizes. 

## 📝 License

This project is intended for educational, research, and portfolio purposes. Please consult clinical experts before using predictive models in real-world medical diagnoses.
