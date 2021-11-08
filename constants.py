import os

# --- Dataset ---
MODES = ['train', 'val', 'test']
DATASET_DIR = 'dataset'
CLASSIFICATION_DATA_DIR = 'numpy_classification_data'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'
TRAIN_LABELS_DIR = 'train_labels'
VAL_LABELS_DIR = 'val_labels'
TEST_LABELS_DIR = 'test_labels'
IMG_RESIZE_DIM = 224

# --- Classification dataset path constants ---
CLASSIFICATION_TRAIN_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TRAIN_DIR)
CLASSIFICATION_TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TRAIN_LABELS_DIR)
CLASSIFICATION_VAL_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, VAL_DIR)
CLASSIFICATION_VAL_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, VAL_LABELS_DIR)
CLASSIFICATION_TEST_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TEST_DIR)
CLASSIFICATION_TEST_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TEST_LABELS_DIR)

# --- Models are saved under {classification/bounding_box}/{model_type}/{model_name}/{epoch}.pth ---
CLASSIFICATION_MODEL_DIR = 'classification_models'
BOUNDING_BOX_MODEL_DIR = 'bounding_box_models'

def get_classification_model_save_dir(model_type, model_name):
    return os.path.join(CLASSIFICATION_MODEL_DIR, model_type, model_name)

def get_bounding_box_model_save_dir(model_type, model_name):
    return os.path.join(BOUNDING_BOX_MODEL_DIR, model_type, model_name)

MODEL_TYPES = [
    'classifier_resnet_18',
]

# --- Training ---
DEFAULT_CLASSIFICATION_EPOCHS = 250
DEFAULT_CLASSIFICATION_LR = 3e-4
GPU = 'cuda:0'

# --- Visuals are saved under {classification/bounding_box}/{model_type}/{model_name}/{img_name}.png ---
CLASSIFICATION_VIZ_DIR = 'classification_viz'
BOUNDING_BOX_VIZ_DIR = 'bounding_box_viz'