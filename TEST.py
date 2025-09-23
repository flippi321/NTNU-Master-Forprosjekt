import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your .nii file
nii_file_path = '/cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/HUNT4/00039/00039_1_SEG_3_PREP_MNI.nii.gz'

# Load the NIfTI image
img = nib.load(nii_file_path)

# Access the image data as a NumPy array
# .get_fdata() returns the image data as a floating-point NumPy array
# .dataobj provides a memory-mapped object, which is useful for large files
image_data = img.get_fdata()

# You can also access the image header and affine transformation matrix
header = img.header
affine = img.affine

# Print some information (optional)
print(f"Image data shape: {image_data.shape}")
print(f"Image data type: {image_data.dtype}")
print(f"Affine transformation matrix:\n{affine}")

# Display view
test = img.get_fdata()[:,:,69]  # Show Random slice
plt.imshow(test)
plt.show()
