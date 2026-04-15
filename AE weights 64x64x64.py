import os
import gc
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_DIR = os.path.join(desktop, "Oulu interpolated")
AE_FOLDER = os.path.join(desktop, "AE")
os.makedirs(AE_FOLDER, exist_ok=True)

# TARGET_SHAPE (64, 64, 64) requires ~2.4x more memory than (48, 48, 48)
TARGET_SHAPE = (64, 64, 64)  
shape_str = f"{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}"

# Dynamic File Naming
H5_FILE = os.path.join(AE_FOLDER, f"fmri_data_{shape_str}.h5")
WEIGHTS_PATH = os.path.join(AE_FOLDER, f"adhd_weights_{shape_str}.pth")
PLOT_PATH = os.path.join(AE_FOLDER, f"loss_plot_{shape_str}.png")

BATCH_SIZE = 4      # Keep low for 64x64x64 stability
LEARNING_RATE = 1e-3         
EPOCHS = 50
PATIENCE = 3 
FILE_PREFIX = 'dswursfMRI_interpolated' 

# ==========================================
# PHASE 1: STREAMING DATA PREPARATION (RAM SAFE)
# ==========================================
def prepare_data_on_desktop():
    if os.path.exists(H5_FILE):
        print(f"Using existing HDF5: {H5_FILE}")
        return

    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.startswith(FILE_PREFIX) and f.endswith('.nii'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"Error: No files found in {DATA_DIR}")
        return

    print(f"Starting memory-safe conversion for {len(all_files)} files at {shape_str}...")
    
    with h5py.File(H5_FILE, 'w') as hf:
        # Create dataset with 'chunks' to optimize disk writing
        dset = hf.create_dataset('volumes', shape=(0, 1, *TARGET_SHAPE), 
                                 maxshape=(None, 1, *TARGET_SHAPE), 
                                 chunks=(1, 1, *TARGET_SHAPE), 
                                 dtype='float32', compression="lzf")

        for path in tqdm(all_files, desc=f"Converting NifTI to H5 ({shape_str})"):
            try:
                # nib.load() only reads the header; it doesn't load the data yet
                img_obj = nib.load(path)
                
                # Determine time points from the header shape
                header_shape = img_obj.header.get_data_shape()
                time_points = header_shape[3] if len(header_shape) == 4 else 1
                
                for t in range(time_points):
                    # Use .dataobj[..., t] to stream ONLY the t-th volume into RAM
                    if time_points > 1:
                        vol = np.asanyarray(img_obj.dataobj[..., t])
                    else:
                        vol = img_obj.get_fdata()
                    
                    # Resize/Normalize
                    factors = [t_s/s for t_s, s in zip(TARGET_SHAPE, vol.shape)]
                    vol_res = zoom(vol, factors, order=1)
                    vol_norm = (vol_res - np.mean(vol_res)) / (np.std(vol_res) + 1e-8)
                    
                    # Append to H5
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = vol_norm[np.newaxis, ...]
                    
                    # Immediate cleanup of the processed volume
                    del vol, vol_res, vol_norm
                
                # Flush the file to disk and force garbage collection
                hf.flush()
                del img_obj
                gc.collect() 
                
            except Exception as e:
                print(f"\nError processing {path}: {e}")

# ==========================================
# PHASE 2: AUTO-SCALING MODELS
# ==========================================
class ADHDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduction factor of 16 (4 layers of stride 2)
        self.flat_dim = (TARGET_SHAPE[0] // 16) 
        self.lin_features = 128 * self.flat_dim**3
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.lin_features, 128), nn.ReLU(),
            nn.Linear(128, 2) 
        )
        self.decoder_fc = nn.Linear(2, self.lin_features)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, self.flat_dim, self.flat_dim, self.flat_dim)),
            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid() 
        )
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(self.decoder_fc(latent)), latent

class DesktopH5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.hf = None 
    def __len__(self):
        with h5py.File(self.path, 'r') as hf:
            return hf['volumes'].shape[0]
    def __getitem__(self, idx):
        if self.hf is None: self.hf = h5py.File(self.path, 'r')
        return torch.from_numpy(self.hf['volumes'][idx])

# ==========================================
# PHASE 3: TRAINING & PLOTTING
# ==========================================
def train():
    prepare_data_on_desktop()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"\n--- Training Started ---")
    print(f"Device: {device} | Resolution: {shape_str} | Batch Size: {BATCH_SIZE}")
    
    model = ADHDAutoencoder().to(device)
    if not os.path.exists(H5_FILE): return
        
    full_dataset = DesktopH5Dataset(H5_FILE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda) 

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf') 
    epochs_no_improve = 0 
    
    master_pbar = tqdm(total=EPOCHS, desc="Overall Progress")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = batch.to(device)
            with torch.amp.autocast('cuda', enabled=use_cuda): 
                recon, _ = model(batch)
                loss = criterion(recon, batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += criterion(recon, batch).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
            
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), WEIGHTS_PATH)
            epochs_no_improve = 0
            status = " [NEW BEST]"
        else:
            epochs_no_improve += 1
            status = f" [{epochs_no_improve}/{PATIENCE}]"

        master_pbar.update(1)
        tqdm.write(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}{status}")
        if epochs_no_improve >= PATIENCE: break
    
    master_pbar.close()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red', linestyle='--')
    plt.title(f'Loss History ({shape_str})')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"\nDone! Weights: {WEIGHTS_PATH}")
    plt.show()

if __name__ == "__main__":
    train()