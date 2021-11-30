import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
sns.set_theme()


def plot_losses_ious(losses_dict, ious_dict, viz_path, prefix='train'):
    '''
    Plots and saves.
    '''
    title_prefix = str.upper(prefix[0]) + prefix[1:]
    
    loss_epochs = list(losses_dict.keys())
    losses = list(losses_dict[x] for x in loss_epochs)
    iou_epochs = list(ious_dict.keys())
    ious = list(ious_dict[x] for x in iou_epochs)

    loss_path = os.path.join(viz_path, prefix + '_losses.png')
    print(f'Saving losses to {loss_path}...')
    loss_plot = sns.lineplot(x=loss_epochs, y=losses)
    plt.title(title_prefix + ' Loss / Epoch')
    plt.ylabel(title_prefix + ' Loss')
    plt.xlabel('Epoch')
    fig = loss_plot.get_figure()
    fig.savefig(loss_path)
    plt.clf()

    iou_path = os.path.join(viz_path, prefix + '_ious.png')
    print(f'Saving ious to {iou_path}...')
    iou_plot = sns.lineplot(x=iou_epochs, y=ious)
    plt.title(title_prefix + ' Iou / Epoch')
    plt.ylabel(title_prefix + ' iou')
    plt.xlabel('Epoch')
    fig = iou_plot.get_figure()
    fig.savefig(iou_path)
    plt.clf()