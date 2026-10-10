import os
import time
import shutil
from pathlib import Path

os.environ["TRUST_REMOTE_CODE"] = "1"
from anomalib.deploy import TorchInferencer

print("--- AUTOMATED OPTICAL INSPECTION SIMULATION ---")

# 1. Setup output sorting directories
output_dir = Path("./conveyor_output")
pass_dir = output_dir / "PASS"
reject_dir = output_dir / "REJECT"

# Clear old folders if they exist, then create fresh ones
if output_dir.exists():
    shutil.rmtree(output_dir)
pass_dir.mkdir(parents=True, exist_ok=True)
reject_dir.mkdir(parents=True, exist_ok=True)

# 2. Load the Model
model_paths = list(Path("./results").rglob("model.pt"))
latest_model = model_paths[-1]
inferencer = TorchInferencer(path=latest_model, device="auto")
print(f"Model loaded on: {inferencer.device}\n")

# 3. Gather all test images (Good and Defective mixed together)
test_dir = Path("./datasets/MVTecAD/bottle/test")
all_images = list(test_dir.rglob("*.png"))
print(f"Starting inspection line. Parts in queue: {len(all_images)}\n")

# 4. Start the Conveyor Belt Loop
stats = {"pass": 0, "reject": 0}

for i, img_path in enumerate(all_images, 1):
    # Start the timer
    start_time = time.time()
    
    # Run inference
    predictions = inferencer.predict(image=str(img_path))
    
    # Stop the timer
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Extract decision
    if hasattr(predictions.pred_label, 'item'):
        is_defective = predictions.pred_label.item()
    else:
        is_defective = predictions.pred_label

    # Trigger the "sorting arm" (copy file to respective folder)
    filename = f"part_{i:03d}_{img_path.name}"
    if is_defective:
        shutil.copy(img_path, reject_dir / filename)
        status = "❌ REJECT"
        stats["reject"] += 1
    else:
        shutil.copy(img_path, pass_dir / filename)
        status = "✅ PASS  "
        stats["pass"] += 1

    # Print the live terminal output
    print(f"Part #{i:03d} | Status: {status} | Latency: {latency_ms:.1f} ms | Original: {img_path.parent.name}")

print("\n--- SHIFT COMPLETE ---")
print(f"Total Parts Inspected: {len(all_images)}")
print(f"Passed: {stats['pass']}")
print(f"Rejected: {stats['reject']}")
