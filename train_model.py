from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

print("1. Loading the dataset...")
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=2,   # Lowered to prevent VRAM overflow
    eval_batch_size=2,
    num_workers=0         # CRITICAL FIX: Disables WSL multiprocessing to prevent teardown crashes
)

print("2. Initializing PatchCore model...")
# Using resnet18 uses significantly less memory and trains much faster
model = Patchcore(backbone="resnet18")

print("3. Setting up the AI Engine...")
engine = Engine(default_root_dir="./results")

print("4. Training the model (Building the memory bank of normal features)...")
try:
    engine.fit(datamodule=datamodule, model=model)
    print("\nSuccess! Training complete.")
    print("The model checkpoints are saved in the './results' folder.")
except Exception as e:
    print(f"\nDetailed Error: {e}")
