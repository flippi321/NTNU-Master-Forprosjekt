import os
import random
import nibabel as nib
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np

class HuntDataLoader():

    def __init__(self, hunts = ['HUNT3', 'HUNT4'], hunt_path = '/cluster/projects/vc/data/mic/closed/MRI_HUNT/images/images_3D_preprocessed/'):
        self.hunts = hunts
        self.hunt_path = hunt_path
        pass

    def get_random_pair(self, verbose=False):
        entry = os.listdir(os.path.join(self.hunt_path, self.hunts[0]))[random.randint(0, len(os.listdir(os.path.join(self.hunt_path, self.hunts[0]))) - 1)]
        
        # Display info regarding the pairs
        if verbose: 
            print("Opening entry:", entry)
        if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], entry)):
            print(f"{entry} exists in both HUNT3 and HUNT4")

            hunt3_path = os.path.join(self.hunt_path, self.hunts[0], entry, entry+'_0_T1_PREP_MNI.nii.gz')
            hunt4_path = os.path.join(self.hunt_path, self.hunts[1], entry, entry+'_1_T1_PREP_MNI.nii.gz')
        else:
            print(f"{entry} does not exist in HUNT4")
            exit()

        return hunt3_path, hunt4_path
    
    def split_training_test_paths(self, split=0.8, seed=random.randint(0, 10000)):
        random.seed(seed)
        all_entries = os.listdir(os.path.join(self.hunt_path, self.hunts[0]))
        random.shuffle(all_entries)
        split_index = int(len(all_entries) * split)
        train_entries = all_entries[:split_index]
        test_entries = all_entries[split_index:]

        train_paths = [(os.path.join(self.hunt_path, self.hunts[0], entry, entry+'_0_T1_PREP_MNI.nii.gz'),
                        os.path.join(self.hunt_path, self.hunts[1], entry, entry+'_1_T1_PREP_MNI.nii.gz')) 
                       for entry in train_entries if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], entry))]
        
        test_paths = [(os.path.join(self.hunt_path, self.hunts[0], entry, entry+'_0_T1_PREP_MNI.nii.gz'),
                       os.path.join(self.hunt_path, self.hunts[1], entry, entry+'_1_T1_PREP_MNI.nii.gz')) 
                      for entry in test_entries if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], entry))]

        return train_paths, test_paths
    
    def load_from_path(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        return data

    def get_middle_slice(self, data_path):
        data = self.load_from_path(data_path)
        return data[:, :, data.shape[2] // 2]

    def get_slice(self, data_path, index):
        data = self.load_from_path(data_path)
        return data[:, :, index]

    def display_slices(self, slice1, slice2):
        # Create figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Show HUNT3 image
        
        axs[0].imshow(slice1, cmap='gray')
        axs[0].set_title('HUNT3 Scan')
        axs[0].axis('off')

        # Show HUNT4 image
        
        axs[1].imshow(slice2, cmap='gray')
        axs[1].set_title('HUNT4 Scan')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    def structural_similarity(self, slice1, slice2):
        data_range = slice1.max() - slice1.min()
        ssim, _ = structural_similarity(slice1, slice2, data_range=data_range, channel_axis=None, full=True)
        return ssim

    def display_slice_differences(self, slice1, slice2, hot=False):
        diff_slice = np.abs(slice1 - slice2)
        plt.figure(figsize=(6, 6))
        plt.axis('off')

        # Display only the differences
        if(hot):
            plt.imshow(diff_slice, cmap='hot')
            plt.title('Differences between HUNT3 and HUNT4')
            plt.colorbar(label='Difference Intensity')

        # Display everything, with differences colored
        else:
            plt.imshow(slice1, cmap='gray')
            plt.imshow(diff_slice, alpha=0.5)
            plt.title('HUNT3 slice with HUNT3↔HUNT4 differences highlighted')
        
        plt.show()
    