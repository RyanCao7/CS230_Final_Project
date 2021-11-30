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


def train_one_epoch(model, train_dataloader, criterion, opt, args):

    # --- Time to train! ---
    total_loss = 0
    total_iou = 0
    total_examples = 0

    model.train()

    for idx, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # --- Move to GPU ---
        # x = x.cuda(constants.GPU, non_blocking=True)
        # y = y.cuda(constants.GPU, non_blocking=True)

        # --- Compute output ---
        output = model(x)
        loss = criterion(output, y)

        # --- Bookkeeping ---
        preds = torch.round(torch.sigmoid(output))

        # --- Metric: IoU, i.e. total number of *correct* positive predictions ---
        # --- out of total number of positive predictions/labels overall ---
        intersection = y * preds
        union = torch.maximum(y, preds)
        union_sum = torch.sum(union, dim=1)
        intersection_sum = torch.sum(intersection, dim=1)

        total_loss += loss.item()
        total_iou += torch.sum(intersection_sum / union_sum).item()
        total_examples += y.shape[0]

        # --- Gradient + GD step ---
        opt.zero_grad()
        loss.backward()
        opt.step()

        avg_loss = total_loss / total_examples
        avg_iou = total_iou / total_examples
        
        if idx % args.print_every_minibatch == 0:
            tqdm.write(f'Train minibatch number: {idx} | Avg loss: {avg_loss} | Avg iou: {avg_iou}')


    return avg_loss, avg_iou, total_examples


def train(args, model, train_dataloader, val_dataloader, criterion, opt):
    """
    Trains and evaluates model.
    """
    print('\n--- Begin training! ---\n')
    train_losses = dict()
    train_ious = dict()
    val_losses = dict()
    val_ious = dict()
    viz_path = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    
    for epoch in range(args.num_epochs):

        if epoch % args.eval_every == 0:
            # --- Time to evaluate! ---
            val_loss, val_iou, _ = eval_model(model, val_dataloader, criterion, args)
            val_losses[epoch] = (val_loss)
            val_ious[epoch] = (val_iou)
            
            # --- Report and plot losses/ious ---
            print(f' Val loss: {val_loss} | Val iou: {val_iou}\n')
            viz_utils.plot_losses_ious(val_losses, val_ious, viz_path, prefix='val')
            
        # --- Train ---
        train_loss, train_iou, _ = train_one_epoch(model, train_dataloader, criterion, opt, args)
        train_losses[epoch] = train_loss
        train_ious[epoch] = train_iou

        # --- Print results ---
        if epoch % args.print_every == 0:
            print(f'Epoch: {epoch} | Train loss: {train_loss} | Train iou: {train_iou}')
        
        # --- Plot loss/ious so far (TODO: do this every epoch?) ---
        # --- Then save all train stats ---
        viz_utils.plot_losses_ious(train_losses, train_ious, viz_path, prefix='train')
        save_train_stats(train_losses, train_ious, val_losses, val_ious, args)
        
        # --- Only save actual model files every so often ---
        if epoch % args.save_every == 0:
            model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch}.pth')
            print(f'Saving model to {model_save_path}...')
            torch.save(model.state_dict(), model_save_path)

    return train_losses, train_ious, val_losses, val_ious


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
            # x = x.cuda(constants.GPU, non_blocking=True)
            # y = y.cuda(constants.GPU, non_blocking=True)

            # --- Compute output ---
            output = model(x)
            loss = criterion(y, output)

            # --- Bookkeeping ---
            # --- TODO(ryancao): What's the metric for multi-label classification? ---
            # --- Metric: IoU, but class-wise ---
            preds = torch.round(torch.sigmoid(output))

            intersection = y * preds
            union = torch.maximum(y, preds)
            union_sum = torch.sum(union, dim=1)
            intersection_sum = torch.sum(intersection, dim=1)

            total_loss += loss.item()
            total_iou += torch.sum(intersection_sum / union_sum).item()
            total_examples += y.shape[0]
            
            # print(output, '\n')
            # print(intersection, intersection.shape, '\n')
            # print(union, union.shape, '\n')
            # print(union_sum, union_sum.shape, '\n')
            # print(intersection_sum, intersection_sum.shape, '\n')
            # exit()

            avg_loss = total_loss / total_examples
            avg_iou = total_iou / total_examples
            if idx % args.print_every_minibatch == 0:
                tqdm.write(f'Val minibatch number: {idx} | Avg loss: {avg_loss} | Avg iou: {avg_iou}')

    return avg_loss, avg_iou, total_examples


def save_train_stats(train_losses, train_ious, val_losses, val_ious, args):
    """Save train stats"""

    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    train_stats = {
        'train_losses': train_losses,
        'train_ious': train_ious,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'model_type': args.model_type,
        'dataset_type': args.dataset_type,
        'train_logits_dir': args.train_logits_dir,
        'train_adv_logits_dir': args.train_adv_logits_dir,
        'val_logits_dir': args.val_logits_dir,
        'val_adv_logits_dir': args.val_adv_logits_dir,
    }
    train_stats_save_path = os.path.join(model_save_dir, 'train_stats.json')
    print(f'Saving current train stats to {train_stats_save_path}...')
    with open(train_stats_save_path, 'w') as f:
        json.dump(train_stats, f)


def main():
    # --- Args ---
    args = opts.get_classify_train_args()
    print('\n' + '-' * 30 + ' Args ' + '-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()
    
    # --- Model and viz save dir ---
    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    viz_save_dir = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
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
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
    print('Done!\n')

    # --- Setup model ---
    # TODO(ryancao): Actually pull the ResNet model! ---
    print('--> Setting up model...')
    model = models.get_model(args.model_type)
    # torch.cuda.set_device(constants.GPU)
    # model = model.cuda(constants.GPU)
    print('Done!\n')

    # --- Optimizer ---
    print('--> Setting up optimizer/criterion...')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Done!\n')
    
    # --- Loss fn ---
    criterion = nn.BCEWithLogitsLoss()#.cuda(constants.GPU)

    # --- Train ---
    train_losses, train_ious, val_losses, val_ious =\
        train(args, model, train_dataloader, val_dataloader, criterion, opt)

    # --- Save model ---
    model_save_path = os.path.join(model_save_dir, 'final_model.pth')
    print(f'Done training! Saving model to {model_save_path}...')
    torch.save(model.state_dict(), model_save_path)
    
    # --- Plot final round of loss/iou metrics ---
    viz_path = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
    viz_utils.plot_losses_ious(train_losses, train_ious, viz_path, prefix='train')
    viz_utils.plot_losses_ious(val_losses, val_ious, viz_path, prefix='val')
    
    # --- Do a final train stats save ---
    save_train_stats(train_losses, train_ious, val_losses, val_ious, args)


if __name__ == '__main__':
    main()