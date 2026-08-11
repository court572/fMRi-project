import nibabel as nib
import numpy as np
import os
from pathlib import Path

# ===== SETTINGS FOR SUB01077 =====
source_file = Path("Z:/Public/Oulu/sub01077/dswursfMRI.nii")
output_dir = Path("Z:/Public/Sinc/sub01077")
output_file = output_dir / "dswursfMRI_interpolated.nii"

old_tr = 1.8
new_tr = 2.0
# =================================

def sinc_interp(data, time_old, time_new):
    """
    Vectorized Sinc Interpolation (Whittaker-Shannon)
    data: 4D array (x, y, z, t)
    """
    T = time_old[1] - time_old[0]
    # Create the interpolation kernel matrix
    sinc_matrix = np.sinc((time_new[:, None] - time_old[None, :]) / T)
    
    # Apply the kernel across the time dimension using dot product
    # Reshapes the 4D data to 2D (voxels, time) for the math, then back to 4D
    shape_orig = data.shape
    data_2d = data.reshape(-1, shape_orig[-1])
    interpolated_2d = np.dot(data_2d, sinc_matrix.T)
    
    new_shape = list(shape_orig[:3]) + [len(time_new)]
    return interpolated_2d.reshape(new_shape)

def run_single_sinc():
    if not source_file.exists():
        print(f"Error: Could not find {source_file}")
        return

    print(f"Loading {source_file.name}...")
    img = nib.load(str(source_file))
    data = img.get_fdata()
    
    # Define timing
    n_trs_old = data.shape[-1]
    time_old = np.arange(n_trs_old) * old_tr
    t_end = time_old[-1]
    time_new = np.arange(0, t_end, new_tr)

    print(f"Performing Sinc Interpolation (TR {old_tr}s -> {new_tr}s)...")
    interp_data = sinc_interp(data, time_old, time_new).astype(np.float32)

    # Update Header
    header = img.header.copy()
    header.set_zooms(list(img.header.get_zooms()[:3]) + [new_tr])
    new_img = nib.Nifti1Image(interp_data, img.affine, header)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, str(output_file))
    
    print(f"Success! File saved to: {output_file}")
    print(f"New volume count: {interp_data.shape[-1]} (was {n_trs_old})")

if __name__ == "__main__":
    run_single_sinc()