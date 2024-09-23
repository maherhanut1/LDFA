from pathlib import Path
import json
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm

import math

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score

from models.FC import FC, FC_rAFA
from models.layers.rAFA_linear import Linear as rAFALinear
from train_cifar_fc_one_cycle import get_cifar10_loaders, get_cifar100_loaders

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_dict_from_json(path: Path):
    with open(path, 'r+') as f:
        loaded_file = json.load(f)
        f.close()
    return loaded_file


def fit_exp(x_points, y_points):
    def exp_func(x, a, b):
        return a * np.exp(b * x)

    # Fit the model to the data
    params, covariance = curve_fit(exp_func, x_points, y_points, sigma=1e-3)
    a, b = params
    return a, b

def _get_acc_array(v, top_k=None):
    vals_acc = []
    for _,v_acc in v.items():
        if top_k is not None:
            v_acc= sorted(v_acc)[-top_k:]
        vals_acc.append(v_acc)
    return np.array(vals_acc) 
    
def plot_accuracies_from_dicts2(acc_dictionaries, top_k=None):
    plt.figure(figsize=(8, 5))
    
    for k,v in acc_dictionaries.items():
        
        vals_acc = _get_acc_array(v, top_k=20)    
        ranks = [int(rank) for rank,_ in v.items()]
        means = vals_acc.mean(1)
        stds = vals_acc.std(1) / np.sqrt(vals_acc.shape[1])
        p_vals = []
        for i in range(vals_acc.shape[0] - 1):
            _, p = ttest_rel(vals_acc[i, :], vals_acc[i+1, :])
            p_vals.append(p)
        
        paired_sorted = sorted(zip(ranks, means, stds, p_vals))
        ranks, means, stds, p_vals = zip(*paired_sorted)
        plt.bar(ranks, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7, error_kw={'capthick': 2}, label='Mean ± STD')

        # Adding labels and title
        plt.xlabel('Time Point', fontsize=12, fontname='Arial')
        plt.ylabel('Mean Value', fontsize=12, fontname='Arial')
        plt.title(f'Bar Plot of Means at Each Time Point, {k}', fontsize=14, fontname='Arial')
        plt.xticks(ranks, fontsize=10, fontname='Arial')
        plt.yticks(fontsize=10, fontname='Arial')
        plt.legend(fontsize=10)

        # Calculate p-values between consecutive time points and annotate
        for i in range(len(means)-1):
            # Annotation for p-values on the plot
            p = p_vals[i]
            plt.text((ranks[i]+ranks[i+1])/2, max(means[i], means[i+1]), f'p={p:.5f}',
                    horizontalalignment='center', color='black')

        plt.tight_layout()
        # Save the plot with high resolution
        plt.savefig('bar_plot_means.png', format='png', dpi=300)
        # Show the plot
        plt.show()
        
    

def plot_accuracies_from_dicts(acc_dictionaries, top_k, save_name, skips=[], extras=[], lims=None):
    
    plt.figure(figsize=(10, 5))
    mpl.style.use('seaborn-whitegrid')
    # Font settings
    plt.rc('font', family='Arial', size=12)
    
    max_rank = 0
    #read accuracies for all layers, all ranks
    blues = plt.cm.Blues(np.linspace(0.4, 1, 4))
    c_idx = 0
    for k,v in acc_dictionaries.items():
        vals_acc = []
        ranks = []
        for r,v_acc in v.items():
            if str(r) in skips:
                continue
            v_acc= sorted(v_acc)[-top_k:]
            vals_acc.append([acc for acc in v_acc])
            ranks.append(int(r))
            # plt.scatter([int(r)]*len(v_acc), v_acc)
            
        #calculate means and stderr
        vals_acc = np.array(vals_acc)
        # ranks = [int(rank) for rank,_ in v.items()]
        max_rank = max(ranks) if max(ranks) > max_rank else max_rank
        means = vals_acc.mean(1)
        stds = vals_acc.std(1)
        paired_sorted = sorted(zip(ranks, means, stds))
        ranks, means, stds = zip(*paired_sorted)
        print(f'for {k} means are {means} for ranks{ranks}, and std are {vals_acc.std(1)}')
        # plt.errorbar(ranks, means, yerr=stds, fmt='-o', capsize=5,
        # capthick=2, ecolor='black', marker='s', 
        # markersize=7, linestyle='-', linewidth=2, label=f'{k}: Mean ± SE')
        means = np.array(means)
        ranks = np.array(ranks)
        stds = np.array(stds)
        
        # plt.errorbar(ranks, (means - means.min()) / (means.max() - means.min()), yerr=stds*0, label=f'{k}: Mean ± SE', fmt='-o', capsize=5)
        plt.errorbar(ranks, means, yerr=stds, label=f'{k}: Mean ± STD', fmt='-o', capsize=5, color=blues[c_idx])
        c_idx+=1
    
    for dictionary, name, color in extras:
        vals_acc =[]
        for _,v_acc in dictionary.items():
            v_acc= sorted(v_acc)[-top_k:]
            vals_acc.append(v_acc)
            vals_acc = np.array(vals_acc)
        vals_acc = np.array(vals_acc)
        mean = vals_acc.mean(1)
        
        plt.plot([1,max_rank], [mean, mean], linestyle='-.', linewidth=2, color=color)
        std = vals_acc.std(1)
        
        plt.errorbar([16], [mean],yerr=[std], fmt='-.', capsize=5,
        capthick=2, ecolor=color, marker='s', 
        markersize=7, linestyle='-.', linewidth=2, label=f'{name}: Mean ± STD', color=color)
        
        
    # plt.grid(True)
    plt.xlabel('Backward Rank', fontsize=18, fontname='Arial')
    plt.ylabel('Accuracy', fontsize=18, fontname='Arial')
    # plt.title('Mean Accuracy and Standard Error over Trials at each Layer', fontsize=14, fontname='Arial')
    plt.xticks(fontsize=15, fontname='Arial')
    plt.yticks(fontsize=15, fontname='Arial')
    if lims is not None:
        plt.ylim(lims)
    plt.legend(fontsize=18)
    plt.savefig(save_name)
    plt.show()
    
    
def plot_accuracies_from_dicts_subsets(subset_names, bn_dicts, const_dicts, skips):
    
    #get bn mean
    bn_means = []
    for bn_dict in bn_dicts:
        for _,v_acc in bn_dict.items():
            bn_means.append(np.mean(v_acc))
            
            
    plt.figure(figsize=(10, 5))
    mpl.style.use('seaborn-whitegrid')
    # Font settings
    plt.rc('font', family='Arial', size=12)
    
    max_rank = 0
    blues = plt.cm.Blues(np.linspace(0.4, 1, 4))
    c_idx = 0
    for i, subset_dict in enumerate(const_dicts):
        
        vals_acc = []
        ranks = []
        for r,v_acc in subset_dict.items():
            
            if str(r) in skips:
                continue
            vals_acc.append([acc for acc in v_acc])
            ranks.append(int(r))
        
        vals_acc = np.array(vals_acc)
        # ranks = [int(rank) for rank,_ in v.items()]
        max_rank = max(ranks) if max(ranks) > max_rank else max_rank
        means = vals_acc.mean(1)
        stds = vals_acc.std(1)
        paired_sorted = sorted(zip(ranks, means, stds))
        ranks, means, stds = zip(*paired_sorted)
        # print(f'for {k} means are {means} for ranks{ranks}, and std are {vals_acc.std(1)}')
        # plt.errorbar(ranks, means, yerr=stds, fmt='-o', capsize=5,
        # capthick=2, ecolor='black', marker='s', 
        # markersize=7, linestyle='-', linewidth=2, label=f'{k}: Mean ± SE')
        means = np.array(means)
        ranks = np.array(ranks)
        stds = np.array(stds)
        
        # plt.errorbar(ranks, (means - means.min()) / (means.max() - means.min()), yerr=stds*0, label=f'{k}: Mean ± SE', fmt='-o', capsize=5)
        plt.errorbar(ranks, means/bn_means[i], yerr=stds, label=f'{subset_names[i]}: Mean ± STD', fmt='-o', capsize=5, color=blues[c_idx])
        c_idx += 1
        
        
    # plt.grid(True)
    plt.xlabel('Backward Rank', fontsize=18, fontname='Arial')
    plt.ylabel('Accuracy', fontsize=18, fontname='Arial')
    # plt.title('Mean Accuracy and Standard Error over Trials at each Layer', fontsize=14, fontname='Arial')
    plt.xticks(fontsize=15, fontname='Arial')
    plt.yticks(fontsize=15, fontname='Arial')
    # if lims is not None:
    #     plt.ylim(lims)
    plt.legend(fontsize=18)
    plt.savefig(rf"to_jonathan/subsets.pdf")
    plt.show()

def evaluate_model(model, testloader):
    val_predictions = []
    val_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs, _ = model(inputs)
            val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
        val_accuracy = accuracy_score(val_targets, val_predictions)
    
    return val_accuracy

def get_accuracies_from_training_path(path, model, layer_idx):
    
    # _, testloader = get_cifar10_loaders(batch_size=64)
    _, testloader = get_cifar10_loaders(batch_size=64)

    path = Path(path)
    accuracies = dict()
    for rank_folder in path.iterdir():
        rank = int(rank_folder.stem.split('_')[-1])
        if rank == 256:
            pass
        accuracies[rank] = []
        for exp_folder in tqdm(rank_folder.iterdir()):
           weights_path = exp_folder / 'checkpoints' / 'best.pth'
           model.net[3] = rAFALinear(512, 512, rank=rank)
           model.net[6] = rAFALinear(512, 512, rank=rank)
           model.net[10] = rAFALinear(512, 10, rank=10)
           model.load_state_dict(torch.load(weights_path))
           model.to('cuda')
           model.eval()
           accuracy = evaluate_model(model, testloader)
           accuracies[rank].append(accuracy)
    with open(path / 'accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        


def create_image_montage(image_list):
    """
    Create a montage of square images.

    Args:
    - image_list: A list of square PIL Image objects.

    Returns:
    - A PIL Image object representing the montage.
    """

    num_images = len(image_list)
    if num_images == 0:
        raise ValueError("No images provided")

    # Determine the number of images per row and column
    num_images_per_row = int(math.ceil(math.sqrt(num_images)))
    num_images_per_col = int(math.ceil(num_images / num_images_per_row))

    # Get the dimensions of the first image
    image_width, image_height = image_list[0].shape

    # Determine the size of the montage
    montage_width = num_images_per_row * image_width
    montage_height = num_images_per_col * image_height

    # Create a new blank image for the montage
    montage = np.zeros((montage_height , montage_width))

    # Paste each image into the montage
    for i in range(num_images):
        row = i // num_images_per_row
        col = i % num_images_per_row
        col_idx = col * image_width
        row_idx = row * image_height
        montage[row_idx:row_idx+image_height, col_idx:col_idx+image_width] = image_list[i]

    return montage


def remove_epochs(path):
    
    path = Path(path)
    for rank_folder in path.iterdir():
        if rank_folder.is_dir():
            for exp_folder in tqdm(rank_folder.iterdir()):
                weights_path = exp_folder / 'checkpoints'
                for file in weights_path.iterdir():
                    if 'final' not in file.name:
                        file.unlink()
            

def plot_cifar10():
    # session_name = 'rAFA_one_layer_128_x4_batch_norm'
    session_name = '512x4_no_drop_out_V2'
    dset= 'cifar10'
    layer_2 = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/layer2/accuracies.json")
    layer_3 = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/layer3/accuracies.json")
    layer_4 = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/layer4/accuracies.json")
    # bp = load_dict_from_json(rf"artifacts/{dset}/{session_name}/all/accuracies.json")
    
    # layer_4_update_p = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/update_P_512_x4_constraint_all_batch_norm_32_32_32/all/accuracies.json")
    layer_2.pop('256')
    layer_3.pop('256')
    layer_2.pop('512')
    layer_3.pop('512')
    layer_2.pop('128')
    layer_3.pop('128')
    layer_2.pop('64')
    layer_3.pop('64')
    layer_2.pop('7')
    layer_3.pop('7')
    layer_2.pop('9')
    layer_3.pop('9')
    BP_baseline = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/BP_decay_1e-6_lr_5e-4/accuracies.json")
    smaller_net = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/64x4_no_constraint/accuracies.json")
    plot_accuracies_from_dicts({'layer1': layer_2, 'layer2': layer_3, 'layer3': layer_4}, top_k=-1, save_name='plots/512_x4.pdf', lims=[0.40, 0.65], extras=[(BP_baseline, 'BP', 'black')]) #]

def plot_width_effect():
    
    net_512_const_all = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/cifar10/512x4_no_drop_out_V2/all_constraint/accuracies.json")
    net_64_no_const = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/cifar10/512x4_no_drop_out_V2/64x4_no_constraint/accuracies.json")
    
    net_64 = np.array(net_64_no_const['64'])
    const_64 = np.array(net_512_const_all['64'])
    const_32 = np.array(net_512_const_all['32'])
    
        
    error_bar_style = {
    'capsize': 5,  # Length of the error bar caps
    'capthick': 2,  # Thickness of the caps
    'elinewidth': 2,  # Line width of the error bars
    'ecolor': 'black'  # Color of the error bars
}
    
    fig, ax = plt.subplots()
    bar_width = 0.25  # Lowering bar width

    bar1 = ax.bar(0, const_64.mean(), yerr=const_64.std(), color='blue', width=bar_width, error_kw=error_bar_style)
    bar2 = ax.bar(0.3, const_32.mean(), yerr=const_32.std(), color='blue', width=bar_width, error_kw=error_bar_style)
    bar3 = ax.bar(0.6, net_64.mean(), yerr=net_64.std(), color='green', width=bar_width, error_kw=error_bar_style)
    ax.legend([bar1, bar3], ['512 neurons', '64 neurons'], loc='upper left')

    
    ax.set_ylim([0.52, None])
    ax.set_ylabel('Accuracy')
    ax.set_xticks([0., 0.3, 0.6])
    ax.set_xticklabels(['rank 32', 'rank 64', 'no constraint'])
    plt.tight_layout()
    plt.savefig('plots/width_effect.pdf')
    plt.show()
    
           
if __name__ == "__main__":
    
    # plot_width_effect()
    # plot_cifar10()
    
    model = FC_rAFA(input_dim=32*32*3, hidden_dim=512, num_classes=10, device='cuda')
    # get_accuracies_from_training_path(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/cifar10/512x4_no_drop_out_V2/all_constraint", model, 3)