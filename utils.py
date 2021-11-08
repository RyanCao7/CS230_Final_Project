import glob
import math
import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

import constants

def compute_train_mean_std():
    """
    Returns the mean/std of the train set.
    """
    running_mean = 0
    num_examples = 0
    print('Starting mean calculation...')
    all_filenames = list(glob.glob(os.path.join(constants.DATASET_DIR, 
                                                 constants.CLASSIFICATION_DATA_DIR, constants.TRAIN_DIR, '*')))
    random.shuffle(all_filenames)
    target_filenames = all_filenames[:5000]
    for file_path in tqdm(target_filenames):
        raw_tensor = torch.from_numpy(np.load(file_path)).double().unsqueeze(0).unsqueeze(0)
        # --- Some tensors have channel values... wtf ---
        if len(raw_tensor.shape) == 5:
            raw_tensor = raw_tensor[:, :, :, :, 0]
        data_tensor = T.Resize((constants.IMG_RESIZE_DIM, constants.IMG_RESIZE_DIM))(raw_tensor)
        running_mean += torch.mean(data_tensor).item()
        num_examples += 1
    train_mean = running_mean / num_examples
    print(f'Got this: train_mean = {train_mean} | running_mean = {running_mean} | num_examples = {num_examples}')

    running_var = 0
    print('Starting std calculation...')
    for file_path in tqdm(target_filenames):
        raw_tensor = torch.from_numpy(np.load(file_path)).double().unsqueeze(0).unsqueeze(0)
        # --- Some tensors have channel values... wtf ---
        if len(raw_tensor.shape) == 5:
            raw_tensor = raw_tensor[:, :, :, :, 0]
        data_tensor = T.Resize((constants.IMG_RESIZE_DIM, constants.IMG_RESIZE_DIM))(raw_tensor)
        running_var += torch.mean(torch.square(data_tensor - train_mean)).item()
    train_std = math.sqrt(running_var / num_examples)
    print(f'Got this: running_var = {running_var} | train_std = {train_std}')
    
    return train_mean, train_std, num_examples


if __name__ == '__main__':
    compute_train_mean_std()