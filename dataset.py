import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import constants


class CXR_Classification_Dataset(Dataset):
    """
    Multi-label classification task for chest x-ray dataset.
    """
    def __init__(self, mode='train'):
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
            print(f'Error: mode should be one of \{train, val, test\} but got {mode} instead.')

        # --- Construct mapping from indices to examples ---
        self.data_filenames = sorted(list(glob.glob(os.path.join(self.data_path, '*'))))
        self.label_filenames = sorted(list(glob.glob(os.path.join(self.label_path, '*'))))

    def __getitem__(self, idx):
        
        # --- Load the numpy array and convert into a torch tensor ---
        x = torch.from_numpy(np.load(self.data_filenames[idx]))
        x.requires_grad = True
        y = torch.from_numpy(np.load(self.label_filenames[idx]))
        y.requires_grad = True

        return x, y

    def __len__(self, idx):
        return len(self.data_filenames)