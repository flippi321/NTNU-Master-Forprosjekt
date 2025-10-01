import os
import torch
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

    def get_pair_path_from_id(self, candidate:str):
        """
        Function which will return the Hunt3 and Hunt4 image paths for a given candidate id
        """
        hunt3_path = os.path.join(self.hunt_path, self.hunts[0], candidate, f'{candidate}_0_T1_PREP_MNI.nii.gz')
        hunt4_path = os.path.join(self.hunt_path, self.hunts[1], candidate, f'{candidate}_1_T1_PREP_MNI.nii.gz')
        return hunt3_path, hunt4_path

    def get_data_info(self, max_entries:int=None):
        """
        Function to print the number of entries, average value for entries and the dimensions of the dataset
        """
        # Get number of entries in each hunt dataset
        hunt3_num = len(os.listdir(os.path.join(self.hunt_path, self.hunts[0])))
        hunt4_num = len(os.listdir(os.path.join(self.hunt_path, self.hunts[1])))
        print(f"Number of entries in {self.hunts[0]}: {hunt3_num}")
        print(f"Number of entries in {self.hunts[1]}: {hunt4_num}")

        # For every candidate we get the MRI pair data
        means_h3 = []
        min_h3_shape = min_h4_shape = [np.inf, np.inf, np.inf]
        max_h3_shape = max_h4_shape = [0, 0, 0]
        means_h4 = []
        for i, candidate in enumerate(os.listdir(os.path.join(self.hunt_path, self.hunts[0]))):
            
            # We load the data
            hunt3 = self.load_from_path(self.get_pair_path_from_id(candidate)[0])
            hunt4 = self.load_from_path(self.get_pair_path_from_id(candidate)[1])

            # Get average
            means_h3.append(np.mean(hunt3))
            means_h4.append(np.mean(hunt4))

            # Get min and max shape
            min_h3_shape = [min(min_h3_shape[0], hunt3.shape[0]), min(min_h3_shape[1], hunt3.shape[1]), min(min_h3_shape[2], hunt3.shape[2])]
            max_h3_shape = [max(max_h3_shape[0], hunt3.shape[0]), max(max_h3_shape[1], hunt3.shape[1]), max(max_h3_shape[2], hunt3.shape[2])]
            min_h4_shape = [min(min_h4_shape[0], hunt4.shape[0]), min(min_h4_shape[1], hunt4.shape[1]), min(min_h4_shape[2], hunt4.shape[2])]
            max_h4_shape = [max(max_h4_shape[0], hunt4.shape[0]), max(max_h4_shape[1], hunt4.shape[1]), max(max_h4_shape[2], hunt4.shape[2])]

            if(max_entries and i >= max_entries):
                break
        
        # Print Average intensity and shape info
        hunt3_mean = np.mean(means_h3)
        hunt4_mean = np.mean(means_h4)
        print(f"Average intensity across Hunt3: {hunt3_mean}")
        print(f"Average intensity across Hunt4: {hunt4_mean}")

        print(f"Min shape across Hunt3: {min_h3_shape}, Max shape across Hunt3: {max_h3_shape}")
        print(f"Min shape across Hunt4: {min_h4_shape}, Max shape across Hunt4: {max_h4_shape}")

        return hunt3_num, hunt4_num, hunt3_mean, hunt4_mean, min_h3_shape, max_h3_shape, min_h4_shape, max_h4_shape

    def get_random_pair(self, verbose:bool=False):
        candidate = os.listdir(os.path.join(self.hunt_path, self.hunts[0]))[random.randint(0, len(os.listdir(os.path.join(self.hunt_path, self.hunts[0]))) - 1)]

        # Display info regarding the pairs
        if verbose:
            print("Viewing candidate:", candidate)
        if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], candidate)):
            print(f"{candidate} exists in both HUNT3 and HUNT4")
            hunt3_path, hunt4_path = self.get_pair_path_from_id(candidate)
        else:
            print(f"{candidate} does not exist in HUNT4")
            exit()

        return hunt3_path, hunt4_path
    
    def split_training_test_paths(self, split=0.8, seed=random.randint(0, 10000)):
        random.seed(seed)
        all_entries = os.listdir(os.path.join(self.hunt_path, self.hunts[0]))
        random.shuffle(all_entries)
        split_index = int(len(all_entries) * split)
        train_entries = all_entries[:split_index]
        test_entries = all_entries[split_index:]

        train_paths = [self.get_pair_path_from_id(candidate)
                       for candidate in train_entries if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], candidate))]

        test_paths = [self.get_pair_path_from_id(candidate)
                      for candidate in test_entries if os.path.exists(os.path.join(self.hunt_path, self.hunts[1], candidate))]

        return train_paths, test_paths
    
    def load_from_path(self, path, crop_size=None):
        img = nib.load(path)
        data = img.get_fdata()

        # If we want to crop the image
        if crop_size:
            if len(crop_size) == 2:  # (H, W) only
                center = np.array(data.shape[:2]) // 2
                start = center - np.array(crop_size) // 2
                end = start + np.array(crop_size)
                data = data[start[0]:end[0], start[1]:end[1], :]
            elif len(crop_size) == 3:  # (H, W, D)
                center = np.array(data.shape) // 2
                start = center - np.array(crop_size) // 2
                end = start + np.array(crop_size)
                data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        return data

    def get_middle_slice(self, data_path):
        data = self.load_from_path(data_path)
        return data[:, :, data.shape[2] // 2]

    def get_slice(self, data_path, index):
        data = self.load_from_path(data_path)
        return data[:, :, index]
    
    def get_all_slices_as_tensor(self, data_path, crop_size=None):
        data = self.load_from_path(data_path, crop_size)
        return [torch.tensor(slice, dtype=torch.float32) for slice in data.transpose(2, 0, 1)]

    def display_slices(self, slice1, slice2, slice1_label='HUNT3 Scan',slice2_label='HUNT4 Scan'):
        # Create figure with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Show HUNT3 image
        
        axs[0].imshow(slice1, cmap='gray')
        axs[0].set_title(slice1_label)
        axs[0].axis('off')

        # Show HUNT4 image
        
        axs[1].imshow(slice2, cmap='gray')
        axs[1].set_title(slice2_label)
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
    
    def to_torch_img(self, x, device):
        """
        x: numpy array or torch tensor with shape (192,224) or (1,192,224), values in [0,1]
        -> returns (1,1,192,224) float32 on device
        """
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = x
        t = t.float()
        if t.ndim == 2:
            t = t.unsqueeze(0)  # (1,H,W)
        elif t.ndim == 3 and t.shape[0] != 1:
            # If it's (H,W,1), move channel first
            if t.shape[-1] == 1:
                t = t.permute(2,0,1)
        t = t.clamp(0, 1)
        t = t.unsqueeze(0)      # (1,1,H,W)
        return t.to(device)

    def to_numpy_img(self, t):
        """
        t: torch tensor (1,1,H,W) or (1,H,W) or (H,W)
        -> numpy (H,W) in [0,1]
        """
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
        arr = t.squeeze().numpy() if isinstance(t, torch.Tensor) else np.array(t).squeeze()
        return np.clip(arr, 0.0, 1.0)
