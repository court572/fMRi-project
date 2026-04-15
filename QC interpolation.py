import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# Hardcoded paths
path_orig = "Z:/Public/Oulu/sub01077"
path_interp = "Z:/Public/Sinc/sub01077"
file_orig = "dswursfMRI.nii"
file_interp = "dswursfMRI_interpolated.nii"

def run_single_subject_qc():
    # 1. Load the images
    orig_fullpath = os.path.join(path_orig, file_orig)
    interp_fullpath = os.path.join(path_interp, file_interp)
    
    img_o = nib.load(orig_fullpath)
    img_i = nib.load(interp_fullpath)
    
    data_o = img_o.get_fdata()
    data_i = img_i.get_fdata()
    
    # 2. Extract central voxel
    mx, my, mz = np.array(data_o.shape[:3]) // 2
    ts_o = data_o[mx, my, mz, :]
    ts_i = data_i[mx, my, mz, :]

    # 3. Setup Time Axes
    tr_o = 1.8
    tr_i = 2.0
    time_o = np.arange(len(ts_o)) * tr_o
    time_i = np.arange(len(ts_i)) * tr_i
    
    # 4. Math Check
    common_time_limit = min(time_o[-1], time_i[-1])
    mask_i_valid = time_i <= common_time_limit
    f_interp = interp1d(time_o, ts_o, kind='cubic')
    ts_o_at_interp_times = f_interp(time_i[mask_i_valid])
    corr, _ = pearsonr(ts_o_at_interp_times, ts_i[mask_i_valid])

    # 5. Visualization
    fig, ax1 = plt.subplots(figsize=(20, 8))
    
    # Plot full length (Removed the 60s zoom_limit)
    ax1.plot(time_o, ts_o, 'r-', label=f'Original (TR={tr_o}s)', alpha=0.3)
    ax1.scatter(time_o, ts_o, color='red', s=10, label='Orig Samples')
    
    ax1.plot(time_i, ts_i, 'b-', label=f'Interpolated (TR={tr_i}s)', linewidth=1.5)
    ax1.scatter(time_i, ts_i, color='blue', marker='x', s=15, label='Interp Samples')

    # Formatting the Timeline
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('BOLD Signal Intensity')
    ax1.set_title(f'Full Scan QC: sub01077 (Duration: ~{common_time_limit/60:.2f} min)\nPearson Correlation: {corr:.5f}')
    
    # Set X-axis limits to full scan
    ax1.set_xlim(0, common_time_limit)

    # Add specific ticks for the TRs (using Locator for clean spacing)
    # Every 1.8s and 2.0s is too dense for labels, but we can show them as grid lines
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10)) # Big labels every 10s
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(tr_o)) # Small tick every 1.8s
    
    # Optional: Create a second axis on top to show the 2.0s grid explicitly
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(20)) # Top labels every 20s
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tr_i)) # Top ticks every 2.0s
    ax2.set_xlabel('Interpolated Timeline (Ticks every 2.0s)')

    ax1.legend(loc='upper right', fontsize='small', ncol=2)
    ax1.grid(True, which='minor', axis='x', color='r', linestyle='--', alpha=0.1) # 1.8s grid
    ax1.grid(True, which='major', axis='both', alpha=0.4)

    plt.tight_layout()
    output_filename = "sub01077_full_scan_QC.png"
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"Full duration plotted: {common_time_limit:.2f} seconds")

if __name__ == "__main__":
    run_single_subject_qc()