import os

# --- Dataset ---
MODES = ['train', 'val', 'test']
DATASET_DIR = 'dataset'
CLASSIFICATION_DATA_DIR = 'numpy_classification_data'
DETECTION_DATA_DIR = 'numpy_bbox_data'
STUDENT_DATA_DIR = 'student_train_data'
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'
TRAIN_LABELS_DIR = 'train_labels'
VAL_LABELS_DIR = 'val_labels'
TEST_LABELS_DIR = 'test_labels'
IMG_RESIZE_DIM = 224

# LABELS_TO_IDXS = {
#     'Cardiomegaly': 0, 
#     'Emphysema': 1, 
#     'Effusion': 2, 
#     'No Finding': 3, 
#     'Hernia': 4, 
#     'Infiltration': 5, 
#     'Mass': 6, 
#     'Nodule': 7, 
#     'Atelectasis': 8, 
#     'Pneumothorax': 9, 
#     'Pleural_Thickening': 10, 
#     'Pneumonia': 11, 
#     'Fibrosis': 12, 
#     'Edema': 13, 
#     'Infiltrate': 5
# }

# --- This comes straight out of the preprocessing script! ---
IDXS_TO_LABELS = {
    0: 'Cardiomegaly',
    1: 'Emphysema',
    2: 'Effusion',
    3: 'No Finding',
    4: 'Hernia',
    5: 'Infiltration',
    6: 'Mass',
    7: 'Nodule',
    8: 'Atelectasis',
    9: 'Pneumothorax',
    10: 'Pleural_Thickening',
    11: 'Pneumonia',
    12: 'Fibrosis',
    13: 'Edema',
    14: 'Consolidation'
}

# --- These weights use "No Finding" as the baseline, as opposed to the total ---
# WEIGHTS = {
#     'Cardiomegaly': 21.743876080691642,
#     'Emphysema': 23.990858505564386,
#     'Effusion': 4.532627468649095,
#     'No Finding': 1.0,
#     'Hernia': 265.9074889867842,
#     'Infiltration': 3.034130893736805,
#     'Mass': 10.439467312348668,
#     'Nodule': 9.534196809350814,
#     'Atelectasis': 5.221991521757937,
#     'Pneumothorax': 11.384571859675594,
#     'Pleural_Thickening': 17.831905465288035,
#     'Pneumonia': 42.18099231306778,
#     'Fibrosis': 35.801304863582445,
#     'Edema': 26.209726443768997,
#     'Consolidation': 12.933576173130492,
# }

# --- These weights are correct, but are hard-capped at 100 ---
WEIGHTS = {
    'No Finding': 1.8574907639038454,
    'Infiltration': 5.6358701115914345,
    'Atelectasis': 9.699801020849554,
    'Effusion': 8.419313659232559,
    'Nodule': 17.70968251461065,
    'Pneumothorax': 21.146737080347037,
    'Mass': 19.391214112763752,
    'Consolidation': 24.023998285836726,
    'Pleural_Thickening': 33.122599704579024,
    'Cardiomegaly': 40.389048991354464,
    'Emphysema': 44.56279809220985,
    'Fibrosis': 66.5005931198102,
    'Edema': 48.68432479374729,
    'Pneumonia': 78.35080363382251,
    'Hernia': 100
}

# def get_indexes_to_labels():
#     """
#     Note: This returns a dict, not a list!
#     """
#     ret = dict()
#     for (label, index) in LABELS_TO_IDXS.items():
#         ret[index] = label
#     return ret

def get_indexes_to_weights():
    """
    Note: This returns a list, not a Tensor!
    """
    weights = list()
    for idx in range(len(IDXS_TO_LABELS)):
        label = IDXS_TO_LABELS[idx]
        weights.append(WEIGHTS[label])
    return weights

# --- Classification dataset path constants ---
CLASSIFICATION_TRAIN_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TRAIN_DIR)
CLASSIFICATION_TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TRAIN_LABELS_DIR)
CLASSIFICATION_VAL_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, VAL_DIR)
CLASSIFICATION_VAL_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, VAL_LABELS_DIR)
CLASSIFICATION_TEST_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TEST_DIR)
CLASSIFICATION_TEST_LABELS_DIR = os.path.join(DATASET_DIR, CLASSIFICATION_DATA_DIR, TEST_LABELS_DIR)

# --- Detection dataset path constants ---
DETECTION_TRAIN_DIR = os.path.join(DATASET_DIR, DETECTION_DATA_DIR, TRAIN_DIR)
DETECTION_TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, DETECTION_DATA_DIR, TRAIN_LABELS_DIR)
DETECTION_VAL_DIR = os.path.join(DATASET_DIR, DETECTION_DATA_DIR, VAL_DIR)
DETECTION_VAL_LABELS_DIR = os.path.join(DATASET_DIR, DETECTION_DATA_DIR, VAL_LABELS_DIR)
STUDENT_TRAIN_DIR = os.path.join(DATASET_DIR, STUDENT_DATA_DIR, TRAIN_DIR)
STUDENT_TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, STUDENT_DATA_DIR, TRAIN_LABELS_DIR)

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
DEFAULT_CLASSIFICATION_LR = 1e-3 # 3e-4 initially
DEFAULT_POS_WEIGHT = 10 # --- How much to weight positive examples over negative ones for each class ---
GPU = 'cuda:0'

# --- Visuals are saved under {classification/bounding_box}/{model_type}/{model_name}/{img_name}.png ---
CLASSIFICATION_VIZ_DIR = 'classification_viz'
BOUNDING_BOX_VIZ_DIR = 'bounding_box_viz'

def get_classification_viz_save_dir(model_type, model_name):
    return os.path.join(CLASSIFICATION_VIZ_DIR, model_type, model_name)

def get_bounding_box_viz_save_dir(model_type, model_name):
    return os.path.join(BOUNDING_BOX_VIZ_DIR, model_type, model_name)