import argparse
import constants
import dataset

def get_classify_train_args():
    """
    Args for `classify_train.py`.
    """
    parser = argparse.ArgumentParser()

    # --- Dataset ---
    parser.add_argument('--train-examples-dir', type=str, help='Train set examples dir', 
                        default=constants.CLASSIFICATION_TRAIN_DIR)
    parser.add_argument('--train-labels-dir', type=str, help='Train set labels dir', 
                        default=constants.CLASSIFICATION_TRAIN_LABELS_DIR)
    parser.add_argument('--val-examples-dir', type=str, help='Val set examples dir', 
                        default=constants.CLASSIFICATION_VAL_DIR)
    parser.add_argument('--val-labels-dir', type=str, help='Val set labels dir', 
                        default=constants.CLASSIFICATION_VAL_LABELS_DIR)

    # --- Model ---
    parser.add_argument('--model-type', type=str, required=True, help=f'Model type. Choices are {constants.MODEL_TYPES}.')

    # --- Hyperparams ---
    parser.add_argument('--lr', type=float, default=constants.DEFAULT_CLASSIFICATION_LR)
    parser.add_argument('--num-epochs', type=int, default=constants.DEFAULT_CLASSIFICATION_EPOCHS)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--pos-weight', type=float, default=constants.DEFAULT_POS_WEIGHT)

    # --- Save dir ---
    parser.add_argument('--model-name', type=str, required=True, help='Where to save model weights')

    # # --- Dataset ---
    # parser.add_argument('--dataset-type', type=str, required=True, 
    #                     help=f'Dataset type. Choices are {dataset.DATASET_TYPES}.',
    #                     default='logits_dataset')
    
    # --- Other ---
    parser.add_argument('--eval-every', type=int, default=1, help='Eval every n epochs.')
    parser.add_argument('--save-every', type=int, default=1, help='Save every n epochs.')
    parser.add_argument('--print-every', type=int, default=1, help='Print every n epochs.')
    parser.add_argument('--print-every-train-minibatch', type=int, default=500, help='Print stats every n train minibatches.')
    parser.add_argument('--print-every-eval-minibatch', type=int, default=100, help='Print stats every n eval minibatches.')
    
    
    args = parser.parse_args()
    return args


def get_classify_eval_args():
    """
    Args for `classify_eval.py`.
    """
    parser = argparse.ArgumentParser()

    # --- Model ---
    parser.add_argument('--model-type', type=str, required=True, help=f'Model type. Choices are {constants.MODEL_TYPES}.')

    # --- Hyperparams ---
    parser.add_argument('--pos-weight', type=float, default=constants.DEFAULT_POS_WEIGHT)

    # --- Save dir ---
    parser.add_argument('--model-name', type=str, required=True, help='Where to load model weights')

    # # --- Dataset ---
    # parser.add_argument('--dataset-type', type=str, required=True, 
    #                     help=f'Dataset type. Choices are {dataset.DATASET_TYPES}.',
    #                     default='logits_dataset')
    
    # --- Other ---
    parser.add_argument('--print-every-minibatch', type=int, default=100, help='Print stats every n minibatches.')
    
    
    args = parser.parse_args()
    return args