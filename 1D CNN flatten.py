import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset

# --- 1. MODEL DEFINITION ---
class TrajectoryAutoencoder(nn.Module):
    def __init__(self):
        super(TrajectoryAutoencoder, self).__init__()
        
        # Encoder: Expects 1 channel, 440 points long
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 440 -> 220
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(5)  # 220 -> 44 (Latent Space)
        )
        
        # Decoder: Rebuilds the 440 points
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5), # 44 -> 220
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),  # 220 -> 440
            nn.Sigmoid() 
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Flatten [Batch, 2, 220] -> [Batch, 1, 440]
        x_flat = x.view(batch_size, 1, 440)
        
        latent = self.encoder(x_flat)
        reconstructed_flat = self.decoder(latent)
        
        # Reshape back to [Batch, 2, 220]
        return reconstructed_flat.view(batch_size, 2, 220)

# --- 2. DATA LOADING & NORMALIZATION (TRANSPOSE ADDED) ---
def load_trajectories(folder_path):
    all_trajectories = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path) # Shape is (220, 2)
        
        # TRANSPOSE: Turn (220, 2) into (2, 220)
        data = data.transpose(1, 0) 
        
        # NORMALIZATION: Scale 0-64 to 0-1
        data = data / 64.0 
        all_trajectories.append(data)
    
    return torch.tensor(np.array(all_trajectories), dtype=torch.float32)

# --- 3. PATH SETUP ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
PATH_TO_DATA = os.path.join(desktop_path, "AE", "Trajectories_FULL_64x64x64")

if not os.path.exists(PATH_TO_DATA):
    print(f"Error: Folder {PATH_TO_DATA} not found.")
    all_data = torch.randn(101, 2, 220) # Dummy backup
else:
    print(f"Success: Loading trajectories from {PATH_TO_DATA}")
    all_data = load_trajectories(PATH_TO_DATA)

# --- 4. PREPARE DATA LOADERS ---
dataset = TensorDataset(all_data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- 5. TRAINING SETUP ---
model = TrajectoryAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200

train_loss_history = []
val_loss_history = []

# --- 6. TRAINING LOOP ---
print("Starting Training...")
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        inputs = batch[0] # inputs shape is now [8, 2, 220]
        optimizer.zero_grad()
        outputs = model(inputs) # outputs shape is [8, 2, 220]
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    train_loss_history.append(running_train_loss / len(train_loader))

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0]
            outputs = model(inputs)
            val_loss = criterion(outputs, inputs)
            running_val_loss += val_loss.item()
            
    val_loss_history.append(running_val_loss / len(val_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_loss_history[-1]:.6f} | Val MSE: {val_loss_history[-1]:.6f}")

# --- 7. PLOT ---
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training Course (Fixed Data Shape)')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()