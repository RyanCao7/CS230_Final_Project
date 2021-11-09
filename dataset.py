import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

import constants


def get_preprocess_transforms():
    """
    Returns a torchvision.transforms composition to apply.
    Note that mean/std were computed via the script within utils.py
    """

    preprocess_transforms = T.Compose([
        T.CenterCrop(256),
        T.Resize(224),
        T.Normalize(
            mean=[129.1120817169278],
            std=[64.12445895568287]
        )
    ])
    
    return preprocess_transforms


class CXR_Classification_Dataset(Dataset):
    """
    Multi-label classification task for chest x-ray dataset.
    """
    def __init__(self, mode='train'):
        self.transform = get_preprocess_transforms()
        if mode == 'train':
            self.data_path = constants.CLASSIFICATION_TRAIN_DIR
            self.label_path = constants.CLASSIFICATION_TRAIN_LABELS_DIR
        elif mode == 'val':
            self.data_path = constants.CLASSIFICATION_VAL_DIR
            self.label_path = constants.CLASSIFICATION_VAL_LABELS_DIR
        elif mode == 'test':
            self.data_path = constants.CLASSIFICATION_TEST_DIR
            self.label_path = constants.CLASSIFICATION_TEST_LABELS_DIR
        else:
            print(f'Error: mode should be one of [train, val, test] but got {mode} instead.')

        # --- Construct mapping from indices to examples ---
        self.data_filenames = sorted(list(glob.glob(os.path.join(self.data_path, '*'))))
        self.label_filenames = sorted(list(glob.glob(os.path.join(self.label_path, '*'))))

    def __getitem__(self, idx):
        
        # --- Load the numpy array and convert into a torch tensor ---
        x = torch.from_numpy(np.load(self.data_filenames[idx])).float().unsqueeze(0)
        # --- Some of the tensors have channels for some reason... ---
        if len(x.shape) == 4:
            x = x[:, :, :, 0]
        x = self.transform(x)
        y = torch.from_numpy(np.load(self.label_filenames[idx])).float()

        return x, y

    def __len__(self):
        return len(self.data_filenames)