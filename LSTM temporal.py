import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
TRAJ_DIR = os.path.join(desktop, "AE", "Trajectories_64x64x64")
LSTM_FOLDER = os.path.join(desktop, "LSTM")
os.makedirs(LSTM_FOLDER, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(LSTM_FOLDER, "lstm_autoencoder_ultra_precision.pth")
PLOT_SAVE_PATH = os.path.join(LSTM_FOLDER, "training_loss_curve.png")

# ULTRA-PRECISION HYPERPARAMETERS
BATCH_SIZE = 4           
EPOCHS = 600             
PATIENCE = 60            
LEARNING_RATE = 5e-5     
HIDDEN_DIM = 256         
NUM_LAYERS = 3           
WEIGHT_DECAY = 1e-6 
VAL_SPLIT = 0.2  # 20% Validation

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class UltraPrecisionLSTM_AE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, num_layers=3):
        super(UltraPrecisionLSTM_AE, self).__init__()
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.2, bidirectional=True)
        
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim * 2, num_layers, 
                               batch_first=True, dropout=0.2)
        
        self.output_layer = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        _, (hidden, _) = self.encoder(x)
        h_top = torch.cat((hidden[-2], hidden[-1]), dim=1) 
        h_repeated = h_top.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(h_repeated)
        return self.output_layer(decoder_out)

# ==========================================
# DATASET & UTILS
# ==========================================
class UnsupervisedTrajDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(data, dtype=torch.float32)

def train_ultra_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset Splitting
    full_dataset = UnsupervisedTrajDataset(TRAJ_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UltraPrecisionLSTM_AE(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"Starting Training on {device}...")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            reconstructed = model(inputs)
            loss = criterion(reconstructed, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                reconstructed = model(inputs)
                loss = criterion(reconstructed, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Early Stopping based on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {avg_train_loss:.8f} | Val MSE: {avg_val_loss:.8f} | Best Val: {best_val_loss:.8f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\n[!] Early stopping at epoch {epoch+1}.")
            break

    # ==========================================
    # FINAL SCORING & VISUALIZATION
    # ==========================================
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    print("\n" + "="*30)
    print(f"FINAL SCORE (Best Val MSE): {best_val_loss:.10f}")
    print("="*30)

    plt.figure(figsize=(12, 7))
    plt.plot(history['train_loss'], label='Training Loss', color='#1f77b4', alpha=0.7)
    plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
    
    best_epoch = history['val_loss'].index(best_val_loss)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label='Best Model Checkpoint')
    
    plt.title(f'LSTM Autoencoder: Training vs Validation\nFinal Score (MSE): {best_val_loss:.8f}', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('MSE (Log Scale)')
    plt.yscale('log') 
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

if __name__ == "__main__":
    train_ultra_ae()