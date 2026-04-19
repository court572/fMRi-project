import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# UPDATE: Point to the FULL folder
TRAJ_DIR = os.path.join(desktop, "AE", "Trajectories_FULL_64x64x64") 

def plot_latent_cloud():
    if not os.path.exists(TRAJ_DIR):
        print(f"Error: Folder not found at {TRAJ_DIR}")
        return

    all_avgs = []
    subject_paths = []
    
    files = [f for f in os.listdir(TRAJ_DIR) if f.endswith("_trajectory.npy")]
    print(f"Gathering coordinates for {len(files)} subjects...")

    for filename in files:
        traj = np.load(os.path.join(TRAJ_DIR, filename)) # Shape: (Time, 2)
        
        # Calculate the average position for the 'Cloud'
        subject_avg = np.mean(traj, axis=0)
        all_avgs.append(subject_avg)
        
        # Store a few full paths to show movement 'traces'
        if len(subject_paths) < 5: 
            subject_paths.append(traj)

    X_avg = np.array(all_avgs)

    if len(X_avg) == 0:
        print("No .npy files found.")
        return

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 1. Density Heatmap (The 'Global' distribution)
    sns.kdeplot(x=X_avg[:, 0], y=X_avg[:, 1], fill=True, cmap="Greens", alpha=0.3, bw_adjust=1.0)
    
    # 2. Individual Subject Averages (The Dots)
    plt.scatter(X_avg[:, 0], X_avg[:, 1], c='darkgreen', s=30, alpha=0.5, label='Subject Centroids')

    # 3. PROOF OF MOTION: Plot the actual trajectories for 5 subjects
    for i, path in enumerate(subject_paths):
        # We plot the path to show it's not a single point
        plt.plot(path[:, 0], path[:, 1], alpha=0.4, lw=1, label=f"Subj {i+1} Path" if i==0 else "")
        # Mark the current frame with a small 'x'
        plt.scatter(path[-1, 0], path[-1, 1], marker='x', s=20, alpha=0.8)

    std_x, std_y = np.std(X_avg[:, 0]), np.std(X_avg[:, 1])
    
    plt.title(f"Latent Space Population Analysis\nSubjects: {len(X_avg)} | Population Spread: X={std_x:.3f}, Y={std_y:.3f}")
    plt.xlabel("Latent Feature 1 (X)")
    plt.ylabel("Latent Feature 2 (Y)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Save to AE folder
    save_path = os.path.join(desktop, "AE", "latent_population_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"Analysis plot saved to: {save_path}")
    plt.show()

    # --- SUCCESS CRITERIA ---
    print("\n--- Thesis Diagnostic ---")
    if std_x > 0.05:
        print("✅ SUCCESS: The model distinguishes between different subjects.")
    
    # Check for movement within a subject
    sample_traj = subject_paths[0]
    intra_std = np.std(sample_traj, axis=0)
    if np.mean(intra_std) > 0.01:
        print(f"✅ SUCCESS: Significant within-subject movement detected (Avg Intra-Std: {np.mean(intra_std):.4f}).")
    else:
        print("⚠️ WARNING: Within-subject movement is very low. Check normalization.")

if __name__ == "__main__":
    plot_latent_cloud()