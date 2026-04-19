import torch
import torch.nn as nn

class SpatioTemporal3DAutoencoder(nn.Module):
    def __init__(self):
        super(SpatioTemporal3DAutoencoder, self).__init__()
        
        # --- ENCODER: Compressing the 3D "Volume" ---
        # Input shape: [Batch, 1, 2, 1, 150] -> (Channels, X, Y, Time)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(2, 1, 3), padding=(0, 0, 1)), # X:2->1, Time:150
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2)), # Reduces Time to 75
            nn.Conv3d(16, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 5))  # Reduces Time to 15
        )
        
        # --- DECODER: Reconstructing the 3D "Volume" ---
        self.decoder = nn.Sequential(
            # Rebuild Time from 15 to 75
            nn.ConvTranspose3d(32, 16, kernel_size=(1, 1, 5), stride=(1, 1, 5)),
            nn.ReLU(),
            # Rebuild Time to 150 and X back to 2
            nn.ConvTranspose3d(16, 1, kernel_size=(2, 1, 2), stride=(2, 1, 2)),
            # No Sigmoid here unless data is scaled 0-1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Training Setup
model_3d = SpatioTemporal3DAutoencoder()
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model_3d.parameters(), lr=0.001)

# Example Input: 10 subjects, 1 channel, 2 (X), 1 (Y), 150 (Time)
input_data_3d = torch.randn(10, 1, 2, 1, 150) 
output_3d = model_3d(input_data_3d)

# Loss: How close is the reconstructed "cube" to the original "cube"?
loss = criterion(output_3d, input_data_3d)