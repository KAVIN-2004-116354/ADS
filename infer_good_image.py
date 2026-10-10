import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from anomalib.deploy import TorchInferencer

# Find the exported model
model_paths = list(Path("./results").rglob("model.pt"))
latest_model = model_paths[-1]

# Initialize Inferencer
inferencer = TorchInferencer(path=latest_model, device="auto")

# ---> CHANGED: Pointing to a known GOOD bottle <---
test_image_path = "./datasets/MVTecAD/bottle/test/good/000.png"
print(f"Running inference on: {test_image_path}")

# Run Prediction
predictions = inferencer.predict(image=test_image_path)

# Extract Data
original_img = cv2.imread(test_image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

heatmap = predictions.anomaly_map.squeeze().cpu().numpy() 
if hasattr(predictions.pred_label, 'item'):
    is_defective = predictions.pred_label.item()
else:
    is_defective = predictions.pred_label

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title(f"Original (Defective: {is_defective})")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Defect Heatmap")
# Setting vmin=0 and vmax=10 keeps the color scale consistent with the broken glass
plt.imshow(heatmap, cmap='jet', vmin=0, vmax=10) 
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Defect Overlay")
plt.imshow(original_img)
plt.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=10)
plt.axis('off')

plt.tight_layout()
output_filename = "good_visualization.png"
plt.savefig(output_filename)
print(f"\nVisualization saved as {output_filename}")
