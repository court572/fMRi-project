import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_DIR = os.path.join(desktop, "Oulu interpolated")

# Set this to match your trained model (48, 48, 48) or (64, 64, 64)
TARGET_SHAPE = (64, 64, 64) 
shape_str = f"{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}"

# Dynamic Paths
OUTPUT_DIR = os.path.join(desktop, "AE", f"Trajectories_{shape_str}")
WEIGHTS_PATH = os.path.join(desktop, "AE", f"adhd_weights_{shape_str}.pth")

TARGET_FILENAME = 'dswursfMRI_interpolated.nii'
BATCH_SIZE = 16  # Higher batch size is fine for 48x48x48

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL (Encoder Only)
# ==========================================
class ADHDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 16 is the reduction factor from 4 layers with stride 2
        flat_dim = TARGET_SHAPE[0] // 16
        lin_features = 128 * (flat_dim ** 3)
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(lin_features, 128), nn.ReLU(),
            nn.Linear(128, 2) 
        )
    def forward(self, x):
        return self.encoder(x)

def run_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Targeting Shape: {shape_str}")

    model = ADHDAutoencoder().to(device)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return
    
    # --- THE CRITICAL FIX: strict=False ---
    # This allows us to load only the Encoder weights from a full Autoencoder file
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model loaded successfully (Encoder weights only).")

    subject_files = []
    for root, _, files in os.walk(DATA_DIR):
        if TARGET_FILENAME in files:
            subject_files.append((os.path.basename(root), os.path.join(root, TARGET_FILENAME)))

    print(f"Found {len(subject_files)} subjects. Starting extraction...")

    with torch.no_grad():
        for sub_id, path in tqdm(subject_files, desc="Extracting"):
            try:
                img_obj = nib.load(path)
                data = torch.from_numpy(img_obj.get_fdata()).float().to(device)
                
                # Reshape to (Time, Channel, D, H, W)
                data = data.permute(3, 0, 1, 2).unsqueeze(1)
                
                # GPU Resizing & Normalization
                data_resized = F.interpolate(data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
                mean = data_resized.mean(dim=(2, 3, 4), keepdim=True)
                std = data_resized.std(dim=(2, 3, 4), keepdim=True)
                data_norm = (data_resized - mean) / (std + 1e-8)

                # Batch Inference to get (x, y) coordinates
                coords_list = []
                for i in range(0, data_norm.size(0), BATCH_SIZE):
                    batch = data_norm[i : i + BATCH_SIZE]
                    coords = model(batch)
                    coords_list.append(coords.cpu().numpy())
                
                trajectory = np.concatenate(coords_list, axis=0)
                np.save(os.path.join(OUTPUT_DIR, f"{sub_id}_trajectory.npy"), trajectory)
                
                # Memory management
                del data, data_resized, data_norm
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error with {sub_id}: {e}")

    print(f"\nSuccess! Trajectories saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_extraction()