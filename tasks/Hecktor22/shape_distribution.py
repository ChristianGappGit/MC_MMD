"""
funtion to compute a distribution of the shapes in the images
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def compute_shape_distribution(img_files, img_path):
    #initialize the arrays
    shapes = np.zeros((len(img_files), 3), dtype=np.int32)
    for i in range(len(img_files)):
        img = nib.load(os.path.join(img_path, img_files[i])).get_fdata().astype(np.float32)
        shapes[i] = img.shape
    return shapes


# Path to the dataset
main_data_path = '/home/christian/data/Hecktor22'
train_data_path = os.path.join(main_data_path, 'hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022')
train_img_path = os.path.join(train_data_path, 'imagesTr_segmented_cropped')

# List images, if they end with .nii.gz
img_files = [f for f in os.listdir(train_img_path) if f.endswith('.nii.gz')]

print(f"Number of images: {len(img_files)}")
print(f"First image: {img_files[0]}")

print("Computing shape distribution...")
shapes = compute_shape_distribution(img_files, train_img_path)

# Plot the shape distribution
plt.hist(shapes[:, 0], bins=10, alpha=0.5, label='x')
plt.hist(shapes[:, 1], bins=10, alpha=0.5, label='y')
plt.hist(shapes[:, 2], bins=10, alpha=0.5, label='z')
plt.legend(loc='upper right')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.title('Shape distribution')
#save the plot
plt.savefig(f'{train_img_path}/shape_distribution.png')
plt.savefig(f'{train_img_path}/shape_distribution.eps')

#Boxplot of shapes in each direction
plt.close()
plt.boxplot(shapes)
plt.xticks([1, 2, 3], ['x', 'y', 'z'])
plt.ylabel('Size')
plt.title('Shape distribution')
#save the plot
plt.savefig(f'{train_img_path}/shape_distribution_boxplot.png')
plt.savefig(f'{train_img_path}/shape_distribution_boxplot.eps')

# DONE
print("DONE")