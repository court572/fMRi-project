import nibabel as nib
import numpy as np
import os
import re
from scipy.interpolate import interp1d
from pathlib import Path

# ===== USER INPUT =====
# The source where the raw data lives on the Z-drive
input_root = Path("Z:/Public/Oulu")

# The destination on the Z-drive
output_root = Path("Z:/Public/Sinc")

old_tr = 1.8   # Original TR
new_tr = 2.0   # Desired TR
# ======================

# Check if Z: drive is accessible
if not os.path.exists("Z:/"):
    print("Error: Z: drive not found. Please ensure 'This PC > Public (Z:)' is connected.")
    exit()

# Create the output folder on the Z-drive
output_root.mkdir(parents=True, exist_ok=True)
print(f"Saving results to: {output_root}")

# Find subject folders (sub01077, etc.)
try:
    subfolders = [
        f for f in input_root.iterdir()
        if f.is_dir() and re.match(r"^sub\d{5}$", f.name)
    ]
except FileNotFoundError:
    print(f"Error: Could not reach {input_root}. Check your network connection.")
    subfolders = []

print(f"Found {len(subfolders)} subject folders in {input_root}.")

for sub_path in subfolders:
    nifti_file_path = sub_path / "dswursfMRI.nii"
    
    if not nifti_file_path.exists():
        print(f"Skipping {sub_path.name}: dswursfMRI.nii not found.")
        continue

    # Define the specific output folder for this subject
    subject_output_dir = output_root / sub_path.name
    output_filename = subject_output_dir / "dswursfMRI_interpolated.nii"

    # Skip if already processed
    if output_filename.exists():
        print(f"Skipping {sub_path.name}: Already exists.")
        continue

    print(f"Processing {sub_path.name}...")

    try:
        # 1. Load the NIfTI
        img = nib.load(str(nifti_file_path))
        data = img.get_fdata()  # (x, y, z, t)
        affine = img.affine
        header = img.header.copy()

        if data.ndim != 4:
            print(f"  Skipping: Data is {data.ndim}D, expected 4D.")
            continue

        # 2. Define time points
        n_trs_old = data.shape[-1]
        time_old = np.arange(n_trs_old) * old_tr
        t_end = time_old[-1]
        time_new = np.arange(0, t_end, new_tr)
        
        # 3. Interpolate across the time axis (last axis)
        # linear interpolation is common for TR alignment
        f = interp1d(time_old, data, axis=-1, kind='linear', fill_value="extrapolate")
        interpolated_data = f(time_new).astype(np.float32)

        # 4. Update metadata
        header.set_zooms(list(img.header.get_zooms()[:3]) + [new_tr])
        new_img = nib.Nifti1Image(interpolated_data, affine, header)
        
        # 5. Save to the new destination
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, str(output_filename))
        
        print(f"  Successfully saved: {sub_path.name}")

    except Exception as e:
        print(f"  Error processing {sub_path.name}: {e}")

print(f"\n--- All files processed! Check the folder: {output_root} ---")