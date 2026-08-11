import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ==========================================
# CONFIGURATION (Server-Side)
# ==========================================
BASE_DIR = "/home/oskla261/MVDRIVE/public"

PATHS_TO_SEARCH = [
    os.path.join(BASE_DIR, "BICENT/ADHD"),
    os.path.join(BASE_DIR, "BICENT/Control")
]

AE_FOLDER = os.path.join(BASE_DIR, "weights_adhd")
TARGET_SHAPE = (64, 64, 64) 

# SYNCED: Pointing to the NEW V2 weights
WEIGHTS_PATH = os.path.join(AE_FOLDER, "adhd_weights_V2_DEEP_2D.pth")

# SYNCED: New folder for high-quality trajectories
OUTPUT_DIR = os.path.join(BASE_DIR, "Trajectories", "Trajectories_V2_Deep_2D")

FILE_PREFIX = 'dswauExam'
BATCH_SIZE = 8 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# SYNCED MODEL: Must match training architecture
# ==========================================
class ADHDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat_dim = 4 
        self.lin_features = 128 * (self.flat_dim**3)
        
        # This matches the 'funnel' logic: ... -> 512 -> 128 -> 2
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.LeakyReLU(0.1),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.1),
            
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.1),
            
            nn.Conv3d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.1),
            
            nn.Flatten(),
            nn.Linear(self.lin_features, 512), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 2) 
        )
        
    def forward(self, x):
        return self.encoder(x)

# ==========================================
# EXTRACTION LOGIC
# ==========================================
def run_extraction():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"🖥️ Using device: {device}")

    model = ADHDAutoencoder().to(device)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ Error: Weights not found at {WEIGHTS_PATH}")
        return
    
    # Load weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    # CHANGED: Added strict=False to ignore decoder weights in the checkpoint
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    subject_files = []
    for group_path in PATHS_TO_SEARCH:
        if not os.path.exists(group_path):
            print(f"⚠️ Path missing: {group_path}")
            continue
        
        group_name = os.path.basename(group_path)
        print(f"🔍 Scanning group: {group_name}")
        
        subs = [d for d in os.listdir(group_path) if d.startswith("Sub")]
        
        for sub in subs:
            pre_path = os.path.join(group_path, sub, "pre")
            if os.path.exists(pre_path):
                for f in os.listdir(pre_path):
                    if f.startswith(FILE_PREFIX) and f.endswith('.nii'):
                        unique_id = f"{group_name}_{sub}" 
                        subject_files.append((unique_id, os.path.join(pre_path, f)))

    if not subject_files:
        print("❌ No subjects found.")
        return

    print(f"🚀 Found {len(subject_files)} subjects. Extracting trajectories...")

    with torch.no_grad():
        for sub_id, path in tqdm(subject_files):
            try:
                img_obj = nib.load(path)
                raw_data = img_obj.get_fdata() 
                
                # Temporal Mean Subtraction
                if len(raw_data.shape) == 4:
                    mean_brain = np.mean(raw_data, axis=3, keepdims=True)
                    functional_data = raw_data - mean_brain
                else:
                    functional_data = raw_data

                data = torch.from_numpy(functional_data).float().to(device)
                
                # Reshape to (Time, Channels, D, H, W)
                if len(data.shape) == 4:
                    data = data.permute(3, 0, 1, 2).unsqueeze(1)
                else:
                    data = data.unsqueeze(0).unsqueeze(0)
                
                # Trilinear interpolation to 64x64x64
                data_resized = F.interpolate(data, size=TARGET_SHAPE, mode='trilinear', align_corners=False)
                
                # Per-volume Z-score normalization
                for t in range(data_resized.size(0)):
                    vol = data_resized[t]
                    data_resized[t] = (vol - vol.mean()) / (vol.std() + 1e-8)

                coords_list = []
                for i in range(0, data_resized.size(0), BATCH_SIZE):
                    batch = data_resized[i : i + BATCH_SIZE]
                    coords = model(batch)
                    coords_list.append(coords.cpu().numpy())
                
                trajectory = np.concatenate(coords_list, axis=0)
                
                # Result saved as e.g., ADHD_Sub01_trajectory.npy
                save_name = f"{sub_id}_trajectory.npy"
                np.save(os.path.join(OUTPUT_DIR, save_name), trajectory)
                
                del data, data_resized
                if use_cuda:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️ Error with {sub_id}: {e}")

    print(f"\n✨ Success! V2 Trajectories saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_extraction()