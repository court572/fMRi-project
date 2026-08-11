import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset

# ==========================================================
# PATH CONFIGURATION
# ==========================================================
BASE_DIR = "/home/oskla261/MVDRIVE/public"

TRAJECTORY_DIR = os.path.join(
    BASE_DIR,
    "Trajectories",
    "Trajectories_adhd_64x64x64"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "Trajectories",
    "trajectory_autoencoder_adhd_weights.pth"
)

PLOT_PATH = os.path.join(
    BASE_DIR,
    "Trajectories",
    "trajectory_autoencoder_adhd_training_loss.png"
)

# ==========================================================
# 1. MODEL DEFINITION (NON-TRIVIAL COMPRESSION Architecture)
# ==========================================================
class TrajectoryAutoencoder(nn.Module):
    def __init__(self):
        super(TrajectoryAutoencoder, self).__init__()

        # Encoder: Extracts hierarchical temporal features
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), # Increased kernel for broader context
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5)
        )

        # Decoder: Reconstructs the clean trajectory from the features
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid() # Restricts output to [0, 1] range matching normalized inputs
        )

    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.size(2) 
        total_elements = 2 * time_steps

        # Flatten the 2 spatial tracks into a 1D sequence
        x_flat = x.view(batch_size, 1, total_elements)

        latent = self.encoder(x_flat)
        reconstructed = self.decoder(latent)

        # Reshape back to the original spatiotemporal format [batch, 2, time_steps]
        reconstructed = reconstructed.view(batch_size, 2, time_steps)

        return reconstructed

# ==========================================================
# 2. LOAD TRAJECTORY DATA
# ==========================================================
def load_trajectories(folder_path):
    trajectories = []

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    if len(files) == 0:
        raise RuntimeError("No trajectory .npy files found.")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        data = np.load(filepath) # Expected shape: (Time, 2)

        # Convert: (Time, 2) -> (2, Time)
        data = data.transpose(1, 0)

        # Spatial normalization: Voxel matrix coordinates (0-64) mapped to [0, 1] range
        data = data / 64.0
        trajectories.append(data)

    return torch.tensor(np.array(trajectories), dtype=torch.float32)

# ==========================================================
# 3. LOAD DATA
# ==========================================================
if not os.path.exists(TRAJECTORY_DIR):
    raise FileNotFoundError(f"\nTrajectory folder not found:\n{TRAJECTORY_DIR}")

print("\nLoading trajectories from:")
print(TRAJECTORY_DIR)

all_data = load_trajectories(TRAJECTORY_DIR)
print(f"Loaded {len(all_data)} trajectories with shape {all_data.shape}.")

# ==========================================================
# 4. DATA SPLIT
# ==========================================================
dataset = TensorDataset(all_data)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ==========================================================
# 5. TRAINING SETUP
# ==========================================================
model = TrajectoryAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200

# Denoising Factor: Standard deviation of the corruption noise
NOISE_FACTOR = 0.05 

train_loss_history = []
val_loss_history = []

best_val_loss = float("inf")
best_epoch = 0

# ==========================================================
# 6. TRAINING LOOP (DENOISING CONFIGURATION)
# ==========================================================
print("\nStarting Denoising Autoencoder Training...\n")

for epoch in range(epochs):
    # --------------------------
    # Training Phase
    # --------------------------
    model.train()
    running_train_loss = 0.0

    for batch in train_loader:
        clean_inputs = batch[0]
        
        # Inject Gaussian noise to break identity lookup shortcuts
        noise = torch.randn_like(clean_inputs) * NOISE_FACTOR
        noisy_inputs = clean_inputs + noise
        
        # Constrain noisy data to valid [0, 1] bounding box bounds
        noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)

        optimizer.zero_grad()
        
        # Predict clean outputs from noisy inputs
        outputs = model(noisy_inputs)
        
        # Loss evaluates reconstruction against the completely CLEAN target trajectory
        loss = criterion(outputs, clean_inputs)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # --------------------------
    # Validation Phase
    # --------------------------
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            clean_inputs = batch[0]
            
            # Validation checks performance on the identical noisy context
            noise = torch.randn_like(clean_inputs) * NOISE_FACTOR
            noisy_inputs = torch.clamp(clean_inputs + noise, 0.0, 1.0)

            outputs = model(noisy_inputs)
            loss = criterion(outputs, clean_inputs)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    # --------------------------
    # Save Best Checkpoint
    # --------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), WEIGHTS_PATH)
        status = " <-- BEST MODEL SAVED"
    else:
        status = ""

    print(
        f"Epoch [{epoch+1:3d}/{epochs}] | "
        f"Train MSE: {avg_train_loss:.6f} | "
        f"Validation MSE: {avg_val_loss:.6f}"
        f"{status}"
    )

# ==========================================================
# 7. SAVE LOSS PLOT
# ==========================================================
plt.figure(figsize=(10,6))
plt.plot(train_loss_history, label="Training Loss (Noisy Input)", linewidth=2)
plt.plot(val_loss_history, label="Validation Loss (Clean Target)", linewidth=2)
plt.title("Denoising Autoencoder Reconstruction Loss Profile", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 8. FINAL SUMMARY
# ==========================================================
print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")
print(f"Best validation MSE: {best_val_loss:.6f}")
print(f"Best epoch: {best_epoch}")
print(f"\nBest robust weights saved to:\n{WEIGHTS_PATH}")
print(f"\nScientific loss plot saved to:\n{PLOT_PATH}")