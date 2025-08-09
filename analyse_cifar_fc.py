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

def load_dict_from_json(path: Path):
    with open(path, 'r+') as f:
        loaded_file = json.load(f)
        f.close()
    return loaded_file


def plot_accuracies_from_dicts(acc_dictionaries, top_k, save_name, skips=[], extras=[], lims=None):
    
    plt.figure(figsize=(8, 4))
    print(mpl.style.available)
    mpl.style.use('seaborn-v0_8-whitegrid')
    
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
        means = np.array(means)
        ranks = np.array(ranks)
        stds = np.array(stds)
        
        # plt.errorbar(ranks, (means - means.min()) / (means.max() - means.min()), yerr=stds*0, label=f'{k}: Mean ± SE', fmt='-o', capsize=5)
        
        if c_idx == 0:
            plt.errorbar(ranks, means, yerr=stds, label=f'{k}', fmt='-o', capsize=5, color=blues[0])
        elif c_idx == 1:
            plt.errorbar(ranks, means, yerr=stds, label=f'{k}', fmt='-o', capsize=5, color=blues[1], linestyle='dashed')
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
        
        plt.errorbar([8], [mean],yerr=[std], fmt='-.', capsize=5,
        capthick=2, ecolor=color, marker='s', 
        markersize=7, linestyle='-.', linewidth=2, label=f'{name}', color=color)
        
        
    # plt.grid(True)
    plt.xlabel('Backward Rank', fontsize=15, fontname='Arial')
    plt.ylabel('Accuracy', fontsize=15, fontname='Arial')
    # plt.title('Mean Accuracy and Standard Error over Trials at each Layer', fontsize=14, fontname='Arial')
    plt.xticks(fontsize=11, fontname='Arial')
    plt.yticks(fontsize=11, fontname='Arial')
    if lims is not None:
        plt.ylim(lims)
    plt.legend(fontsize=18)
    plt.savefig(save_name)
    plt.savefig(save_name.split('.')[0] + '.SVG')
    plt.show()
    
    
def plot_accuracies_from_dicts_subsets(subset_names, bn_dicts, const_dicts, skips, lims):
    
    #get bn mean
    bn_means = []
    for bn_dict in bn_dicts:
        for _,v_acc in bn_dict.items():
            bn_means.append(np.mean(v_acc))
            
            
    plt.figure(figsize=(8, 4))
    mpl.style.use('seaborn-v0_8-whitegrid')
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
        if c_idx == 0:
            plt.errorbar(ranks, means, yerr=stds, label=f'{subset_names[i]}', fmt='-o', capsize=5, color=blues[c_idx])
            plt.plot([45, 128], [bn_means[i], bn_means[i]], color='black', linewidth=1.5)
        if c_idx == 1:
            plt.errorbar(ranks, means, yerr=stds, label=f'{subset_names[i]}', fmt='-o', capsize=5, color=blues[c_idx], linestyle='dotted')
            plt.plot([50, 128], [bn_means[i], bn_means[i]], linestyle='dotted', color='black', linewidth=1.5)
        if c_idx == 2:
            plt.errorbar(ranks, means, yerr=stds, label=f'{subset_names[i]}', fmt='-o', capsize=5, color=blues[c_idx], linestyle='dashed')
            plt.plot([55, 128], [bn_means[i], bn_means[i]], linestyle='dashed', color='black', linewidth=1.5)
            
        c_idx += 1
        
        
    # plt.grid(True)
    plt.xlabel('Backward Rank', fontsize=15, fontname='Arial')
    plt.ylabel('Accuracy', fontsize=15, fontname='Arial')
    # plt.title('Mean Accuracy and Standard Error over Trials at each Layer', fontsize=14, fontname='Arial')
    plt.xticks(fontsize=11, fontname='Arial')
    plt.yticks(fontsize=11, fontname='Arial')
    # if lims is not None:
    plt.ylim(lims)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(rf"plots/subsets_layer4_2.SVG")
    plt.savefig(rf"plots/subsets_layer4_2.pdf")
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
    # _, testloader = get_cifar10_loaders(batch_size=64)
    _, testloader = get_cifar100_loaders(batch_size=64, class_limit=100)

    path = Path(path)
    accuracies = dict()
    for rank_folder in path.iterdir():
        rank = int(rank_folder.stem.split('_')[-1])
        accuracies[rank] = []
        for exp_folder in tqdm(rank_folder.iterdir()):
            weights_path = exp_folder / 'checkpoints' / 'best.pth'
            # model.net[layer_idx] = rAFALinear(512, 100, rank=rank)
        #    model.net[6] = rAFALinear(512, 512, rank=rank)
        #    model.net[10] = rAFALinear(512, 10, rank=10)
            model.load_state_dict(torch.load(weights_path), strict=False)
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

def plot_cifar_subsets():
    
    bp_50 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/50/BP/accuracies.json")
    bp_75 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/75/BP/accuracies.json")
    bp_100 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/100/BP/accuracies.json")
    
    layer2_50 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/50/layer2/accuracies.json")
    layer2_75 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/75/layer2/accuracies.json")
    layer2_100 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/100/layer2/accuracies.json")
    
    layer4_50 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/50/layer4/accuracies.json")
    layer4_75 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/75/layer4/accuracies.json")
    layer4_100 = load_dict_from_json(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/100/layer4/accuracies.json")
    
    const_all_50 = load_dict_from_json(rf'artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/50/Constraint_all/accuracies.json')
    const_all_75 = load_dict_from_json(rf'artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/75/constraint_all/accuracies.json')
    const_all_100 = load_dict_from_json(rf'artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/100/Constraint_all/accuracies.json')
    
    layer2_50.pop('42')
    layer2_75.pop('64')
    layer2_75.pop('70')
    layer2_75.pop('75')


    plot_accuracies_from_dicts_subsets(subset_names=['50 Classes', '75 Classes', '100 Classes'], bn_dicts=[bp_50, bp_75, bp_100], const_dicts=[const_all_50, const_all_75, const_all_100], skips=[], lims=[None, None])
    # plot_accuracies_from_dicts_subsets(subset_names=[50, 75, 100], bn_dicts=[bp_50], const_dicts=[layer4_50], skips=[], lims=[None, None])

def plot_cifar10():
    
    layer_2 = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_lr_6e4_wd_4e-4_gamma_975_update_QP/layer2/accuracies.json")
    layer_3 = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_lr_6e4_wd_4e-4_gamma_975_update_QP/layer3/accuracies.json")
    layer_4 = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_lr_6e4_wd_4e-4_gamma_975_update_QP/layer4/accuracies.json")
    BP_baseline = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_no_drop_out_BP_5e-4_decay_4e-4_gamma0.975/all/accuracies.json")

    # smaller_net = load_dict_from_json(rf"/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/{dset}/{session_name}/64x4_no_constraint/accuracies.json")
    layer_2.pop('64')
    layer_3.pop('64')
    layer_2.pop('32')
    layer_3.pop('32')
    plot_accuracies_from_dicts({'$\mathregular{B_1}$': layer_2, '$\mathregular{B_2}$': layer_3, '$\mathregular{B_3}$': layer_4}, top_k=10, save_name='plots/512_x4.pdf', lims=[0.43, 0.65], extras=[(BP_baseline, 'BP', 'black')]) #]


def plot_DFA():
    
    DFA = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_DFA_lr_6e-4_decay_4e-4_gamma_975/layer4/accuracies.json")
    DFA_prev = load_dict_from_json(rf'artifacts/cifar10/512_x4_all/DFA_from_layer_3_lr_6e4_wd_4e4/layer4/accuracies.json')
    BP_baseline = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_no_drop_out_BP_5e-4_decay_4e-4_gamma0.975/all/accuracies.json")
    DFA_prev.pop('32')
    plot_accuracies_from_dicts({'last layer': DFA, 'penultimate layer': DFA_prev}, top_k=10, save_name='plots/512_x4_DFA.pdf', lims=[0.3, 0.65], extras=[(BP_baseline, 'BP', 'black')]) #]


def plot_width_effect():
    
    
    blues = plt.cm.Blues(np.linspace(0.65, 0.75, 4))
    greens = plt.cm.Greens(np.linspace(0.65, 0.75, 4))
    
    net_512_const_all = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_lr_6e4_wd_4e-4_gamma_975_update_QP_constrain_all/constraint_all/accuracies.json")
    net_64_no_const = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/64x4_no_drop_out_BP_5e-4_decay_4e-4_gamma0.975/all/accuracies.json")
    BP_baseline = load_dict_from_json(rf"artifacts/cifar10/512_x4_all/512x4_no_drop_out_BP_5e-4_decay_4e-4_gamma0.975/all/accuracies.json")
    
    
    BP = np.array(BP_baseline['512'])
    net_64 = np.array(net_64_no_const['64'])
    
    const_64 = np.array(net_512_const_all['64'])
    const_32 = np.array(net_512_const_all['32'])
    const_16 = np.array(net_512_const_all['16'])
    const_10 = np.array(net_512_const_all['10'])
    
        
    error_bar_style = {
    'capsize': 5,  # Length of the error bar caps
    'capthick': 2,  # Thickness of the caps
    'elinewidth': 2,  # Line width of the error bars
    'ecolor': 'black'  # Color of the error bars
}
    
    fig, ax = plt.subplots(figsize=(7,4))
    bar_width = 0.37  # Lowering bar width
    space = 0.4
    bar0 = ax.bar(0, BP.mean(), yerr=BP.std(), color='black', width=bar_width, error_kw=error_bar_style)
    bar1 = ax.bar(space * 1, const_64.mean(), yerr=const_64.std(), color=blues[0], width=bar_width, error_kw=error_bar_style)
    bar2 = ax.bar(space * 2, const_32.mean(), yerr=const_32.std(), color=blues[1], width=bar_width, error_kw=error_bar_style)
    bar3 = ax.bar(space * 3, const_16.mean(), yerr=const_16.std(), color=blues[2], width=bar_width, error_kw=error_bar_style)
    bar4 = ax.bar(space * 4, const_10.mean(), yerr=const_10.std(), color=blues[3], width=bar_width, error_kw=error_bar_style)
    bar5 = ax.bar(space * 5, net_64.mean(), yerr=net_64.std(), color=greens[[0]], width=bar_width, error_kw=error_bar_style)
    
    # ax.legend([bar0, bar1, bar5], ['512 neurons', '512 neurons', '64 neurons'], loc='upper right', fontsize=14)

    
    ax.set_ylim([0.55, 0.65])
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.set_xticks([space * i for i in range(6)])
    ax.set_xticklabels(['BP', 'rank 64', 'rank 32', 'rank 16', 'rank 10', '64 neurons'], fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/width_effect.pdf')
    plt.savefig('plots/width_effect.SVG')
    plt.show()
    plt.close()
    
           
if __name__ == "__main__":
    # plot_DFA()
    plot_cifar_subsets()
    # plot_width_effect()
    # plot_cifar10()
    
    
    # # model = FC(input_dim=32*32*3, hidden_dim=512, num_classes=10, device='cuda')
    # model = FC(input_dim=32*32*3, hidden_dim=512, num_classes=100, device='cuda')
    # get_accuracies_from_training_path(rf"artifacts/cifar_subsets/512x4_lr_6e-4_wd_4e-4_gamma_975_subsets/100/Constraint_all", model, 10)