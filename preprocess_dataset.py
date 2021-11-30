import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

import constants


NUM_VAL_FILES = 5000
TRAIN_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.TRAIN_DIR)
TRAIN_LABELS_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.TRAIN_LABELS_DIR)
VAL_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.VAL_DIR)
VAL_LABELS_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.VAL_LABELS_DIR)
TEST_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.TEST_DIR)
TEST_LABELS_WRITE_DIR = os.path.join(constants.DATASET_DIR, constants.CLASSIFICATION_DATA_DIR, constants.TEST_LABELS_DIR)


def process_labels():
    """
    Goes through labels in data CSV file and assigns an index to each one.
    
    Step 1: image_labels: dict(image_name --> [c_1, c_2, ..., c_k]) (maps to class indices)
    Step 2: image_label_encodings: dict(image_name --> "one hot" label encodings)
    """

    # --- Image names have indices associated with them ---
    image_names_to_idx = dict()
    idx_to_image_names = dict()

    # --- Labels have indices associated with them ---
    idx_to_labels = dict()
    labels_to_idx = dict()
    current_idx = 0
    
    # --- Read in labels ---
    image_labels = dict()
    with open(os.path.join(constants.DATASET_DIR, 'Data_Entry_2017.csv'), 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                print(line)
            else:

                # --- Store filenames ---
                image_filename = line.split(',')[0]
                image_names_to_idx[image_filename] = line_idx - 1
                idx_to_image_names[line_idx - 1] = image_filename

                # --- Read in labels ---
                labels = line.split(',')[1].split('|')
                label_indexes = list()
                for label in labels:
                    # --- Update label indices (have to convert from string --> idx) ---
                    if label not in labels_to_idx:
                        labels_to_idx[label] = current_idx
                        idx_to_labels[current_idx] = label
                        current_idx += 1
                    label_indexes.append(labels_to_idx[label])
                image_labels[image_filename] = label_indexes
    
    # --- Convert labels to "one-hot' encodings ---
    image_label_encodings = dict()
    for (image_name, label_indexes) in image_labels.items():
        encoding = np.zeros(len(idx_to_labels))
        encoding[label_indexes] = 1
        image_label_encodings[image_name] = encoding
    
    # --- Finally, stack all the encodings into one giant numpy array ---
    stacked_labels = list()
    for i in range(len(idx_to_image_names)):
        stacked_labels.append(image_label_encodings[idx_to_image_names[i]])

    stacked_labels = np.stack(stacked_labels)
        
    return image_names_to_idx, idx_to_image_names, idx_to_labels, labels_to_idx, image_labels, stacked_labels


def preprocess_train_val_test_data():
    """
    Idea: convert all the PNG files to .npy files.
    Let's do 5000 val/remainder train.
    """
    
    # --- Read in train/val data filenames ---
    train_val_filenames = list()
    with open(os.path.join(constants.DATASET_DIR, 'train_val_list.txt'), 'r') as f:
        for line in f:
            train_val_filenames.append(line.strip())
    
    # --- Read in test data filenames ---
    test_filenames = list()
    with open(os.path.join(constants.DATASET_DIR, 'test_list.txt'),  'r') as f:
        for line in f:
            test_filenames.append(line.strip())
    test_filenames = set(test_filenames)
    
    # --- Shuffle and split ---
    random.shuffle(train_val_filenames)
    train_filenames = set(train_val_filenames[5000:])
    val_filenames = set(train_val_filenames[:5000])

    # --- Process image labels ---
    image_names_to_idx, idx_to_image_names, idx_to_labels, labels_to_idx, image_labels, stacked_labels = process_labels()
    
    print('Index to labels:')
    print(idx_to_labels)

    # --- Go through all images in all image directories ---
    for image_dir in sorted(glob.glob(os.path.join(constants.DATASET_DIR, 'images_0*'))):
        print(f'We are going through {image_dir} now...')
        for image_path in tqdm(glob.glob(os.path.join(image_dir, 'images', '*'))):
            image_name = os.path.basename(image_path)
            image = Image.open(image_path)
            image_arr = np.asarray(image)
            label_arr = stacked_labels[image_names_to_idx[image_name]]
            save_name = image_name[:-len('.png')]
            if image_name in train_filenames:
                np.save(os.path.join(TRAIN_WRITE_DIR, save_name), image_arr)
                np.save(os.path.join(TRAIN_LABELS_WRITE_DIR, save_name), label_arr)
            elif image_name in val_filenames:
                np.save(os.path.join(VAL_WRITE_DIR, save_name), image_arr)
                np.save(os.path.join(VAL_LABELS_WRITE_DIR, save_name), label_arr)
            elif image_name in test_filenames:
                np.save(os.path.join(TEST_WRITE_DIR, save_name), image_arr)
                np.save(os.path.join(TEST_LABELS_WRITE_DIR, save_name), label_arr)
            else:
                print(f'Oh God. We can\'t find {image_path} with name {image_name} in either list :(')
    

if __name__ == '__main__':
    preprocess_train_val_test_data()