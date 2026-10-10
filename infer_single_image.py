import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from anomalib.deploy import TorchInferencer

# 1. Find the exported Torch model (.pt file)
model_paths = list(Path("./results").rglob("model.pt"))
if not model_paths:
    print("Error: No exported model.pt found in ./results.")
    exit()
latest_model = model_paths[-1]
print(f"Loading inference model from: {latest_model}")

# 2. Initialize the Inferencer
inferencer = TorchInferencer(path=latest_model, device="auto")

# 3. Select a defective test image
test_image_path = "./datasets/MVTecAD/bottle/test/broken_large/000.png"
print(f"Running inference on: {test_image_path}")

# 4. Run the prediction
predictions = inferencer.predict(image=test_image_path)

# 5. Extract the results
original_img = cv2.imread(test_image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# ---> THE FIX: Move tensors from GPU to CPU, then convert to NumPy <---
heatmap = predictions.anomaly_map.squeeze().cpu().numpy() 

# Safely extract the boolean value whether it's a tensor or a standard bool
if hasattr(predictions.pred_label, 'item'):
    is_defective = predictions.pred_label.item()
else:
    is_defective = predictions.pred_label

# 6. Visualize the Results using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title(f"Original (Defective: {is_defective})")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Defect Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Defect Overlay")
plt.imshow(original_img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis('off')

plt.tight_layout()

output_filename = "defect_visualization.png"
plt.savefig(output_filename)
print(f"\nVisualization saved as {output_filename}")
