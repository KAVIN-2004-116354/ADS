import matplotlib.pyplot as plt
from anomalib.data import MVTecAD
import cv2
import numpy as np

print("Initializing MVTec AD Dataset module...")

# Removed the image_size argument
datamodule = MVTecAD(
    root="./datasets/MVTecAD", 
    category="bottle",
    train_batch_size=1,
    eval_batch_size=1
)

print("Downloading and preparing data (this might take a few minutes)...")
datamodule.prepare_data()
datamodule.setup()

print("Data successfully downloaded and loaded!")
print(f"Number of training images (Normal only): {len(datamodule.train_data)}")
print(f"Number of testing images (Normal + Anomalies): {len(datamodule.test_data)}")
