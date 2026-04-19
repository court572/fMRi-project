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

TARGET_SHAPE = (64, 64, 64) 
shape_str = f"{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}"

# UPDATED: Paths to match your new FULL training run
OUTPUT_DIR = os.path.join(desktop, "AE", f"Trajectories_FULL_{shape_str}")
WEIGHTS_PATH = os.path.join(desktop, "AE", f"adhd_weights_FULL_{shape_str}.pth")

TARGET_FILENAME = 'dswursfMRI_interpolated.nii'
BATCH_SIZE = 8  # Adjusted for 64x64x64 stability

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL (Encoder Only - Matches Full Robust)
# ==========================================
class ADHDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        flat_dim = TARGET_SHAPE[0] // 16
        lin_features = 128 * (flat_dim ** 3)
        
        # Added BatchNorm3d to match the new training architecture
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(),
            
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
            
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(lin_features, 128), nn.ReLU(),
            nn.Linear(128, 2) 
        )
    def forward(self, x):
        return self.encoder(x)

def run_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ADHDAutoencoder().to(device)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return
    
    # Load weights with strict=False to ignore the Decoder weights in the file
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Full Robust Model loaded successfully.")

    subject_files = []
    for root, _, files in os.walk(DATA_DIR):
        if TARGET_FILENAME in files:
            subject_files.append((os.path.basename(root), os.path.join(root, TARGET_FILENAME)))

    print(f"Starting extraction for {len(subject_files)} subjects...")

    with torch.no_grad():
        for sub_id, path in tqdm(subject_files):
            try:
                img_obj = nib.load(path)
                raw_data = img_obj.get_fdata() # (W, H, D, T)
                
                # --- CRITICAL: TEMPORAL MEAN SUBTRACTION ---
                # Matches the logic used in Phase 1 of training
                if len(raw_data.shape) == 4:
                    mean_brain = np.mean(raw_data, axis=3, keepdims=True)
                    functional_data = raw_data - mean_brain
                else:
                    functional_data = raw_data

                data = torch.from_numpy(functional_data).float().to(device)
                
                # Reshape to (Time, Channel, D, H, W)
                data = data.permute(3, 0, 1, 2).unsqueeze(1)
                
                # GPU Resizing
                data_resized = F.interpolate(data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
                
                # --- CRITICAL: VOLUME-WISE STANDARDIZATION ---
                # Matches Phase 1: (vol - mean) / std
                for t in range(data_resized.size(0)):
                    vol = data_resized[t]
                    data_resized[t] = (vol - vol.mean()) / (vol.std() + 1e-8)

                # Batch Inference to get (x, y) coordinates
                coords_list = []
                for i in range(0, data_resized.size(0), BATCH_SIZE):
                    batch = data_resized[i : i + BATCH_SIZE]
                    coords = model(batch)
                    coords_list.append(coords.cpu().numpy())
                
                trajectory = np.concatenate(coords_list, axis=0)
                np.save(os.path.join(OUTPUT_DIR, f"{sub_id}_trajectory.npy"), trajectory)
                
                # Explicit cleanup to keep GPU memory free
                del data, data_resized
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error with {sub_id}: {e}")

    print(f"\nSuccess! Full trajectories saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_extraction()