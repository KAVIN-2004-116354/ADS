from pathlib import Path
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 1. Automatically find the latest saved model checkpoint
checkpoints = list(Path("./results").rglob("*.ckpt"))
if not checkpoints:
    print("Error: No checkpoints found in ./results")
    exit()
latest_checkpoint = checkpoints[-1]
print(f"Found checkpoint: {latest_checkpoint}")

# 2. Load the dataset (We only need the test set now)
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    eval_batch_size=2,
    num_workers=0
)

# 3. Initialize the model and engine
model = Patchcore(backbone="resnet18")
engine = Engine()

# 4. Run the test phase
print("Testing model against defective images...")
engine.test(model=model, datamodule=datamodule, ckpt_path=latest_checkpoint)
