import os
import torch

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_unet_model.pth")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Training datasets
TRAIN_DATASETS = {
    "Kvasir-SEG": {
        "images": os.path.join(DATASET_DIR, "Kvasir-SEG", "images"),
        "masks":  os.path.join(DATASET_DIR, "Kvasir-SEG", "masks"),
    },
    "CVC-ClinicDB": {
        "images": os.path.join(DATASET_DIR, "CVC-ClinicDB", "images"),
        "masks":  os.path.join(DATASET_DIR, "CVC-ClinicDB", "masks"),
    },
}

# Evaluation-only datasets
EVAL_DATASETS = {
    "CVC-ColonDB": {
        "images": os.path.join(DATASET_DIR, "CVC-ColonDB", "images"),
        "masks":  os.path.join(DATASET_DIR, "CVC-ColonDB", "masks"),
    },
    "CVC-300": {
        "images": os.path.join(DATASET_DIR, "CVC-300", "images"),
        "masks":  os.path.join(DATASET_DIR, "CVC-300", "masks"),
    },
    "EndoSceneStill": {
        "images": os.path.join(DATASET_DIR, "EndoSceneStill", "TestDataset", "images"),
        "masks":  os.path.join(DATASET_DIR, "EndoSceneStill", "TestDataset", "masks"),
    },
}

# -----------------------------
# Hyperparameters
# -----------------------------

IMAGE_SIZE = 352
BATCH_SIZE = 4
EPOCHS = 80
LR = 1e-4

# -----------------------------
# Device
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
