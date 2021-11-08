import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
sns.set_theme()


def plot_losses_accuracies(losses_dict, accuracies_dict, viz_path, prefix='train'):
    '''
    Plots and saves.
    '''
    title_prefix = str.upper(prefix[0]) + prefix[1:]
    
    loss_epochs = list(losses_dict.keys())
    losses = list(losses_dict[x] for x in loss_epochs)
    accuracy_epochs = list(accuracies_dict.keys())
    accuracies = list(accuracies_dict[x] for x in accuracy_epochs)

    loss_path = os.path.join(viz_path, prefix + '_losses.png')
    print(f'Saving losses to {loss_path}...')
    loss_plot = sns.lineplot(x=loss_epochs, y=losses)
    plt.title(title_prefix + ' Loss / Epoch')
    plt.ylabel(title_prefix + ' Loss')
    plt.xlabel('Epoch')
    fig = loss_plot.get_figure()
    fig.savefig(loss_path)
    plt.clf()

    acc_path = os.path.join(viz_path, prefix + '_accuracies.png')
    print(f'Saving accuracies to {acc_path}...')
    acc_plot = sns.lineplot(x=accuracy_epochs, y=accuracies)
    plt.title(title_prefix + ' Accuracy / Epoch')
    plt.ylabel(title_prefix + ' Acc')
    plt.xlabel('Epoch')
    fig = acc_plot.get_figure()
    fig.savefig(acc_path)
    plt.clf()