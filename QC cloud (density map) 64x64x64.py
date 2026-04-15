import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
# Ensure this matches your extraction folder name exactly
TRAJ_DIR = os.path.join(desktop, "AE", "Trajectories_64x64x64") 

def plot_latent_cloud():
    if not os.path.exists(TRAJ_DIR):
        print(f"Error: Folder not found at {TRAJ_DIR}")
        return

    all_coords = []
    
    print("Gathering subject coordinates...")
    for filename in os.listdir(TRAJ_DIR):
        if filename.endswith("_trajectory.npy"):
            # Load the (Time, 2) array
            traj = np.load(os.path.join(TRAJ_DIR, filename))
            
            # Method: Average each subject into a single (x, y) point
            subject_avg = np.mean(traj, axis=0)
            all_coords.append(subject_avg)

    X = np.array(all_coords)

    if len(X) == 0:
        print("No .npy files found in the directory.")
        return

    # ==========================================
    # PLOTTING THE CLOUD
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 1. Draw the "Density" (The Cloud)
    sns.kdeplot(x=X[:, 0], y=X[:, 1], fill=True, cmap="Blues", alpha=0.5, bw_adjust=0.8)
    
    # 2. Draw the individual subjects (The Dots)
    plt.scatter(X[:, 0], X[:, 1], c='navy', s=25, alpha=0.6, edgecolors='white', label='Healthy Subjects')

    # Calculate spread metrics for the console
    std_x, std_y = np.std(X[:, 0]), np.std(X[:, 1])
    
    plt.title(f"Latent Space 'Cloud' Analysis\nSubjects: {len(X)} | Spread (Std): X={std_x:.3f}, Y={std_y:.3f}")
    plt.xlabel("Latent Feature 1 (X)")
    plt.ylabel("Latent Feature 2 (Y)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Save to Desktop/AE
    plt.savefig(os.path.join(desktop, "AE", "latent_cloud_check.png"), dpi=300)
    print(f"Cloud plot saved to AE folder.")
    plt.show()

    # --- SUCCESS CRITERIA CHECK ---
    print("\n--- Diagnostic Check ---")
    if std_x < 0.01 and std_y < 0.01:
        print("⚠️ WARNING: COLLAPSE DETECTED. All subjects are in the same spot.")
    else:
        print("✅ SUCCESS: The model sees differences between subjects.")

if __name__ == "__main__":
    plot_latent_cloud()