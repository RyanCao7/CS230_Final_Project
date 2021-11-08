import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm

import constants
import dataset
import models
import opts
import utils
import viz_utils


def get_preprocess_transforms():
    """
    Returns a torchvision.transforms composition to apply.
    Note that mean/std were computed via the script within utils.py
    """

    preprocess_transforms = T.Compose([
       T.Resize(224),
       T.Normalize(
           mean=[129.1120817169278],
           std=[64.12445895568287]
       )
    ])
    
    return preprocess_transforms


def train_one_epoch(model, train_dataloader, criterion, opt):

    # --- Time to train! ---
    total_loss = 0
    total_correct = 0
    total_examples = 0

    model.train()

    for idx, (x, y) in tqdm(enumerate(train_dataloader)):

        # --- Move to GPU ---
        x = x.cuda(constants.GPU, non_blocking=True)
        y = y.cuda(constants.GPU, non_blocking=True)

        # --- Compute output ---
        output = model(x)
        loss = criterion(output, y)

        # --- Bookkeeping ---
        preds = torch.round(output)
        total_loss += loss.item()
        total_correct += torch.sum(preds == y).item()
        total_examples += y.shape.numel()

        # --- Gradient + GD step ---
        opt.zero_grad()
        loss.backward()
        opt.step()

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc, total_examples


def train(args, model, train_dataloader, val_dataloader, criterion, opt):
    """
    Trains and evaluates model.
    """
    print('\n--- Begin training! ---\n')
    train_losses = dict()
    train_accuracies = dict()
    val_losses = dict()
    val_accuracies = dict()
    
    for epoch in range(args.num_epochs):

        if epoch % args.eval_every == 0:
            # --- Time to evaluate! ---
            val_loss, val_acc, _ = eval_model(model, val_dataloader, criterion)
            val_losses[epoch] = (val_loss)
            val_accuracies[epoch] = (val_acc)
            print(f' Val loss: {val_loss} | Val acc: {val_acc}\n')

        # --- Train ---
        train_loss, train_acc, _ = train_one_epoch(model, train_dataloader, criterion, opt)
        train_losses[epoch] = train_loss
        train_accuracies[epoch] = train_acc

        # --- Print results ---
        if epoch % args.print_every == 0:
            print(f'Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}')

    return train_losses, train_accuracies, val_losses, val_accuracies


def eval_model(model, val_dataloader, criterion):
    '''
    Evaluates model over validation set.
    '''
    print('\n' + ('-' * 30) + ' Evaluating model! ' + ('-' * 30))
    total_loss = 0
    total_correct = 0
    total_examples = 0

    model.eval()

    with torch.no_grad():
        for idx, (x, y) in tqdm(enumerate(val_dataloader)):

            # --- Move to GPU ---
            x = x.cuda(constants.GPU, non_blocking=True)
            y = y.cuda(constants.GPU, non_blocking=True)

            # --- Compute output ---
            output = model(x)
            loss = criterion(output, y)

            # --- Bookkeeping ---
            # TODO(ryancao): What's the metric for multi-label classification? ---
            preds = torch.round(output)
            total_loss += loss.item()
            total_correct += torch.sum(preds == y).item()
            total_examples += y.shape.numel()

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples

    return avg_loss, avg_acc, total_examples


def main():
    # --- Args ---
    args = opts.get_classify_train_args()
    print('\n' + '-' * 30 + ' Args ' + '-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()
    
    # --- Model and viz save dir ---
    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    viz_save_dir = get_bounding_box_model_save_dir(args.model_type, args.model_name)
    if os.path.isdir(model_save_dir):
        raise RuntimeError(f'Error: {model_save_dir} already exists! Exiting...')
    elif os.path.isdir(viz_save_dir):
        raise RuntimeError(f'Error: {viz_save_dir} already exists! Exiting...')
    else:
        print(f'--> Creating directory {model_save_dir}...')
        os.makedirs(model_save_dir)
        print(f'--> Creating directory {viz_save_dir}...')
        os.makedirs(viz_save_dir)
    print('Done!\n')

    # --- Setup dataset ---
    print('--> Setting up dataset...')
    train_dataset = dataset.CXR_Classification_Dataset(mode='train')
    val_dataset = dataset.CXR_Classification_Dataset(mode='val')
    print('Done!\n')

    # --- Dataloaders ---
    print('--> Setting up dataloaders...')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
    print('Done!\n')

    # --- Setup model ---
    # TODO(ryancao): Actually pull the ResNet model! ---
    print('--> Setting up model...')
    model = models.get_model(args.model_type)
    torch.cuda.set_device(constants.GPU)
    model = model.cuda(constants.GPU)
    print('Done!\n')

    # --- Optimizer ---
    print('--> Setting up optimizer/criterion...')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Done!\n')
    
    # --- Loss fn ---
    # TODO(ryancao): Is this correct for multi-label training?
    criterion = nn.BCELoss().cuda(constants.GPU)
    
    # --- Train ---
    train_losses, train_accuracies, val_losses, val_accuracies =\
        train(args, model, train_dataloader, val_dataloader, criterion, opt)

    # --- Save model ---
    model_save_path = os.path.join(model_save_dir, 'final_model.pth')
    print(f'Done training! Saving model to {model_save_path}...')
    torch.save(model.state_dict(), model_save_path)
    viz_utils.plot_losses_accuracies(train_losses, train_accuracies, viz_path, prefix='train')
    viz_utils.plot_losses_accuracies(val_losses, val_accuracies, viz_path, prefix='val')
    
    # --- Save train stats ---
    train_stats = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'model_type': args.model_type,
        'dataset_type': args.dataset_type,
        'train_logits_dir': args.train_logits_dir,
        'train_adv_logits_dir': args.train_adv_logits_dir,
        'val_logits_dir': args.val_logits_dir,
        'val_adv_logits_dir': args.val_adv_logits_dir,
    }
    train_stats_save_path = os.path.join(model_save_dir, 'train_stats.json')
    print(f'Saving train stats to {train_stats_save_path}...')
    with open(train_stats_save_path, 'w') as f:
        json.dump(train_stats, f)

if __name__ == '__main__':
    main()