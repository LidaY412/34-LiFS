import os
from Nii_utils import NiiDataRead, NiiDataWrite
import numpy as np
from skimage import transform

target_size = (32, 160, 192)
values_clip = (-55, 145)
data_dir = "./data/val/"
save_dir = "./data/preprocessed_val/"

ids = os.listdir(data_dir)

for ID in ids:
    os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
    for mode in ['1', '2', '3', '4', '5', '6', '7']:
        file_path = os.path.join(data_dir, ID, f'raw data{mode}.nii.gz')

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Create an all-zero placeholder image")
            img = np.zeros(target_size, dtype=np.float32)
            spacing = (1.0, 1.0, 1.0)
            origin = (0.0, 0.0, 0.0)
            direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        else:
            img, spacing, origin, direction = NiiDataRead(file_path)

            if len(img.shape) == 4:
                img = img[:, :, :, 0]

            img = np.clip(img, values_clip[0], values_clip[1])
            img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

        spacing_z = (spacing[0] * img.shape[0]) / target_size[0]
        spacing_x = (spacing[1] * img.shape[1]) / target_size[1]
        spacing_y = (spacing[2] * img.shape[2]) / target_size[2]

        if not os.path.exists(file_path):
            new_spacing = np.array([spacing_z, spacing_x, spacing_y])
        else:
            img = transform.resize(img, target_size, order=0, mode='constant',
                                   clip=False, preserve_range=True, anti_aliasing=False)

            if img.shape != tuple(target_size):
                print(f"Dimension mismatch after resizing for {ID}, mode {mode}. Skipping...")
                continue
            
            new_spacing = np.array([spacing_z, spacing_x, spacing_y])

        NiiDataWrite(os.path.join(save_dir, ID, '{}_img.nii.gz'.format(mode)), img,
                     new_spacing, origin, direction)

os.makedirs("./relevant_files/", exist_ok=True)
with open("./relevant_files/val.txt", "w") as f:
    for id in ids:
        f.write(f"{id}\n")