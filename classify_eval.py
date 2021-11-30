import glob
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

import constants
import dataset
import models
import opts
import utils
import viz_utils


def eval_model(model, val_dataloader, criterion, args):
    '''
    Evaluates model over validation set.
    '''
    print('\n' + ('-' * 30) + ' Evaluating model! ' + ('-' * 30))
    total_loss = 0
    total_iou = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():
        for idx, (x, y) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

            # --- Move to GPU ---
            x = x.cuda(constants.GPU, non_blocking=True)
            y = y.cuda(constants.GPU, non_blocking=True)

            # --- Compute logits ---
            logits = model(x)
            loss = criterion(y, logits)

            # --- Bookkeeping ---
            # --- TODO(ryancao): What's the metric for multi-label classification? ---
            # --- Metric: IoU, but class-wise ---
            preds = torch.round(torch.sigmoid(logits))

            intersection = y * preds
            union = torch.maximum(y, preds)
            union_sum = torch.sum(union, dim=1)
            intersection_sum = torch.sum(intersection, dim=1)

            total_loss += loss.item()
            total_iou += torch.sum(intersection_sum / union_sum).item()
            total_examples += y.shape[0]

            avg_loss = total_loss / total_examples
            avg_iou = total_iou / total_examples
            if idx % args.print_every_minibatch == 0:
                tqdm.write(f'Val minibatch number: {idx} | Avg loss: {avg_loss} | Avg iou: {avg_iou}')

    return avg_loss, avg_iou, total_examples


def main():
    # --- Args ---
    args = opts.get_classify_eval_args()
    print('\n' + '-' * 30 + ' Args ' + '-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()
    
    # --- Model and viz save dir ---
    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    viz_save_dir = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
    if not os.path.isdir(model_save_dir):
        raise RuntimeError(f'Error: {model_save_dir} does not exist! Exiting...\n')
    
    # --- Get model weight path ---
    model_filenames = sorted(list(os.path.basename(x) for x in glob.glob(os.path.join(model_save_dir, '*.pth'))))
    for model_file in model_filenames:
        print(model_file)
    user_choice = input('Please select which model .pth file to load -> ')
    while user_choice not in model_filenames:
        user_choice = input (f'Error. Failed to find specified model. Please try again -> ')
    model_weights_path = os.path.join(model_save_dir, user_choice)
    print(f'--> Selected {model_weights_path} as the model weights file.\n')

    # --- Setup dataset ---
    print('--> Setting up dataset...')
    val_dataset = dataset.CXR_Classification_Dataset(mode='train')
    print('Done!\n')

    # --- Dataloaders ---
    print('--> Setting up dataloaders...')
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
    print('Done!\n')

    # --- Setup model ---
    # TODO(ryancao): Actually pull the ResNet model! ---
    print('--> Setting up model...')
    model = models.get_model(args.model_type)
    model.load_state_dict(torch.load(model_weights_path))
    torch.cuda.set_device(constants.GPU)
    model = model.cuda(constants.GPU)
    print('Done!\n')
    
    # --- Loss fn ---
    pos_weight = torch.Tensor(constants.get_indexes_to_weights())
    print(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda(constants.GPU)

    # --- Run eval ---
    avg_loss, avg_iou, total_examples = eval_model(model, val_dataloader, criterion, args)
    print(f'\nFinal loss: {avg_loss} | Final iou: {avg_iou} | Total number of val examples: {total_examples}')


if __name__ == '__main__':
    main()