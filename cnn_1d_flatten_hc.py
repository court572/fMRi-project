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
    "Trajectories_V2_hc_2d"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "Trajectories",
    "trajectory_autoencoder_hc_weights.pth"
)

PLOT_PATH = os.path.join(
    BASE_DIR,
    "Trajectories",
    "trajectory_autoencoder_training_loss.png"
)


# ==========================================================
# 1. MODEL DEFINITION
# ==========================================================

class TrajectoryAutoencoder(nn.Module):

    def __init__(self):
        super(TrajectoryAutoencoder, self).__init__()

        # Encoder
        # Input: [batch, 1, 440]

        self.encoder = nn.Sequential(

            nn.Conv1d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),
            # 440 -> 220


            nn.Conv1d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(5)
            # 220 -> 44
        )


        # Decoder

        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(
                32,
                16,
                kernel_size=5,
                stride=5
            ),

            nn.ReLU(),

            nn.ConvTranspose1d(
                16,
                1,
                kernel_size=2,
                stride=2
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        batch_size = x.size(0)

        # [batch,2,220] -> [batch,1,440]

        x_flat = x.view(
            batch_size,
            1,
            440
        )


        latent = self.encoder(x_flat)


        reconstructed = self.decoder(latent)


        # [batch,1,440] -> [batch,2,220]

        reconstructed = reconstructed.view(
            batch_size,
            2,
            220
        )


        return reconstructed



# ==========================================================
# 2. LOAD TRAJECTORY DATA
# ==========================================================

def load_trajectories(folder_path):

    trajectories = []


    files = sorted(
        [
            f for f in os.listdir(folder_path)
            if f.endswith(".npy")
        ]
    )


    if len(files) == 0:

        raise RuntimeError(
            "No trajectory .npy files found."
        )


    for filename in files:


        filepath = os.path.join(
            folder_path,
            filename
        )


        # Original:
        # (220,2)

        data = np.load(filepath)


        # Convert:
        # (220,2) -> (2,220)

        data = data.transpose(
            1,
            0
        )


        # Normalize coordinates
        # 0-64 -> 0-1

        data = data / 64.0


        trajectories.append(data)



    return torch.tensor(
        np.array(trajectories),
        dtype=torch.float32
    )



# ==========================================================
# 3. LOAD DATA
# ==========================================================

if not os.path.exists(TRAJECTORY_DIR):

    raise FileNotFoundError(
        f"\nTrajectory folder not found:\n{TRAJECTORY_DIR}"
    )


print(
    "\nLoading trajectories from:"
)

print(
    TRAJECTORY_DIR
)


all_data = load_trajectories(
    TRAJECTORY_DIR
)


print(
    f"Loaded {len(all_data)} trajectories."
)



# ==========================================================
# 4. DATA SPLIT
# ==========================================================

dataset = TensorDataset(
    all_data
)


train_size = int(
    0.8 * len(dataset)
)


val_size = len(dataset) - train_size



train_dataset, val_dataset = random_split(
    dataset,
    [
        train_size,
        val_size
    ]
)



train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)



# ==========================================================
# 5. TRAINING SETUP
# ==========================================================

model = TrajectoryAutoencoder()


criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 200



train_loss_history = []

val_loss_history = []


best_val_loss = float("inf")

best_epoch = 0



# ==========================================================
# 6. TRAINING LOOP
# ==========================================================

print("\nStarting Training...\n")


for epoch in range(epochs):


    # --------------------------
    # Training
    # --------------------------

    model.train()


    running_train_loss = 0.0



    for batch in train_loader:


        inputs = batch[0]


        optimizer.zero_grad()


        outputs = model(inputs)


        loss = criterion(
            outputs,
            inputs
        )


        loss.backward()


        optimizer.step()


        running_train_loss += loss.item()



    avg_train_loss = (
        running_train_loss /
        len(train_loader)
    )


    train_loss_history.append(
        avg_train_loss
    )



    # --------------------------
    # Validation
    # --------------------------

    model.eval()


    running_val_loss = 0.0



    with torch.no_grad():


        for batch in val_loader:


            inputs = batch[0]


            outputs = model(inputs)


            loss = criterion(
                outputs,
                inputs
            )


            running_val_loss += loss.item()



    avg_val_loss = (
        running_val_loss /
        len(val_loader)
    )


    val_loss_history.append(
        avg_val_loss
    )



    # --------------------------
    # Save best weights
    # --------------------------

    if avg_val_loss < best_val_loss:


        best_val_loss = avg_val_loss


        best_epoch = epoch + 1


        torch.save(
            model.state_dict(),
            WEIGHTS_PATH
        )


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

plt.figure(
    figsize=(10,6)
)


plt.plot(
    train_loss_history,
    label="Training Loss",
    linewidth=2
)


plt.plot(
    val_loss_history,
    label="Validation Loss",
    linewidth=2
)


plt.title(
    "Training and Validation Reconstruction Loss",
    fontsize=14
)


plt.xlabel(
    "Epoch",
    fontsize=12
)


plt.ylabel(
    "Mean Squared Error (MSE)",
    fontsize=12
)


plt.grid(
    True,
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ==========================================================
# 8. FINAL SUMMARY
# ==========================================================

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    f"Best validation MSE: {best_val_loss:.6f}"
)

print(
    f"Best epoch: {best_epoch}"
)


print(
    "\nBest weights saved to:"
)

print(
    WEIGHTS_PATH
)


print(
    "\nLoss plot saved to:"
)

print(
    PLOT_PATH
)