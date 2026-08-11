import os
import gc
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
BASE_DIR = "/home/oskla261/MVDRIVE/public"
PATH_ADHD = os.path.join(BASE_DIR, "BICENT/ADHD")
PATH_CONTROL = os.path.join(BASE_DIR, "BICENT/Control")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_npy_cache")
AE_FOLDER = os.path.join(BASE_DIR, "weights_adhd")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(AE_FOLDER, exist_ok=True)

TARGET_SHAPE = (64, 64, 64)
LATENT_DIM = 2

WEIGHTS_PATH = os.path.join(AE_FOLDER, "adhd_weights_V2_DEEP_2D.pth")
PLOT_PATH = os.path.join(AE_FOLDER, "adhd_weight_plot_V2.png")
SUMMARY_PATH = os.path.join(AE_FOLDER, "training_summary_V2.txt")

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 150
PATIENCE = 15
FILE_PREFIX = 'dswauExam'

# ==========================================
# PHASE 1: DATA EXTRACTION
# ==========================================
def prepare_data_as_npy():
    existing = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('.npy')]
    if len(existing) > 0:
        print(f"✅ Found {len(existing)} cached volumes. Skipping extraction.")
        return

    all_files = []
    for group_path in [PATH_ADHD, PATH_CONTROL]:
        if not os.path.exists(group_path):
            continue
        print(f"🔍 Scanning group: {os.path.basename(group_path)}")
        for item in os.listdir(group_path):
            sub_folder = os.path.join(group_path, item)
            if os.path.isdir(sub_folder) and item.startswith("Sub"):
                pre_folder = os.path.join(sub_folder, "pre")
                if os.path.exists(pre_folder):
                    for f in os.listdir(pre_folder):
                        if f.startswith(FILE_PREFIX) and f.endswith('.nii'):
                            all_files.append(os.path.join(pre_folder, f))

    print(f"🚀 Processing {len(all_files)} subjects...")

    global_idx = 0
    for path in tqdm(all_files):
        try:
            img_obj = nib.load(path)
            full_data = img_obj.get_fdata()

            time_points = full_data.shape[3] if len(full_data.shape) == 4 else 1

            subject_mean = np.mean(full_data, axis=3, keepdims=True) if time_points > 1 else 0
            functional_data = full_data - subject_mean if time_points > 1 else full_data[..., np.newaxis]

            for t in range(time_points):
                vol = functional_data[..., t]
                factors = [t_s / s for t_s, s in zip(TARGET_SHAPE, vol.shape)]
                vol_res = zoom(vol, factors, order=1)
                vol_norm = (vol_res - np.mean(vol_res)) / (np.std(vol_res) + 1e-8)

                save_path = os.path.join(PROCESSED_DATA_DIR, f"vol_{global_idx:06d}.npy")
                np.save(save_path, vol_norm.astype(np.float32))
                global_idx += 1

            gc.collect()

        except Exception as e:
            print(f"⚠️ Error: {e}")

# ==========================================
# DATASET
# ==========================================
class NpyVolumeDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.file_list = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.folder, self.file_list[idx]))
        return torch.from_numpy(data).unsqueeze(0)

# ==========================================
# MODEL
# ==========================================
class ADHDAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.flat_dim = 4
        self.lin_features = 128 * (self.flat_dim ** 3)

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),

            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.Conv3d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
            nn.Linear(self.lin_features, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, LATENT_DIM)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, self.lin_features),
            nn.LeakyReLU(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, self.flat_dim, self.flat_dim, self.flat_dim)),

            nn.ConvTranspose3d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose3d(32, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(self.decoder_fc(latent))
        return recon, latent

# ==========================================
# TRAINING
# ==========================================
def train():
    prepare_data_as_npy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    full_dataset = NpyVolumeDataset(PROCESSED_DATA_DIR)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ADHDAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    actual_epochs_trained = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            with torch.amp.autocast('cuda'):
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

        history['train'].append(avg_train)
        history['val'].append(avg_val)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        actual_epochs_trained = epoch + 1

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), WEIGHTS_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("🏁 Early stopping triggered.")
                break

    # ==========================================
    # FINAL PLOTS + SUMMARY
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(history['train'], label='Training Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    initial_train_loss = history['train'][0]
    final_train_loss = history['train'][-1]

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Initial training loss : {initial_train_loss:.6f}")
    print(f"Final training loss   : {final_train_loss:.6f}")
    print(f"Best validation loss  : {best_val_loss:.6f}")
    print(f"Epoch of best model   : {best_epoch}")
    print(f"Total epochs trained  : {actual_epochs_trained}")
    print("=" * 60)

    with open(SUMMARY_PATH, "w") as f:
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Initial training loss : {initial_train_loss:.6f}\n")
        f.write(f"Final training loss   : {final_train_loss:.6f}\n")
        f.write(f"Best validation loss  : {best_val_loss:.6f}\n")
        f.write(f"Epoch of best model   : {best_epoch}\n")
        f.write(f"Total epochs trained  : {actual_epochs_trained}\n")

    print(f"✅ Weights saved to: {WEIGHTS_PATH}")
    print(f"📈 Loss plot saved to: {PLOT_PATH}")
    print(f"📄 Summary saved to: {SUMMARY_PATH}")

if __name__ == "__main__":
    train()