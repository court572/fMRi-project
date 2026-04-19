import torch
import torch.nn as nn

class TrajectoryAutoencoder(nn.Module):
    def __init__(self):
        super(TrajectoryAutoencoder, self).__init__()
        
        # --- ENCODER: Squeezes the signal ---
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # Time 150 -> 75
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(5)  # Time 75 -> 15 (This is your 'Latent' compression)
        )
        
        # --- DECODER: Rebuilds the signal ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5), # 15 -> 75
            nn.ReLU(),
            nn.ConvTranspose1d(16, 2, kernel_size=2, stride=2),  # 75 -> 150
            nn.Sigmoid() # Keeps output in a specific range if data is normalized
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Training Setup
model = TrajectoryAutoencoder()
criterion = nn.MSELoss() # Measures the difference between original and rebuild
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example: 10 healthy brains
input_data = torch.randn(10, 2, 150)
output = model(input_data)

# Training Loss calculation:
loss = criterion(output, input_data)