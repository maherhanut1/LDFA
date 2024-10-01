from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import math

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from models.alexnet import AlexNet_cifar, AlexNet_AFA
from models.layers.rAFA_conv import Conv2d as AFAConv2d

def load_dict_from_json(path: Path):
    with open(path, 'r+') as f:
        loaded_file = json.load(f)
        f.close()
    return loaded_file


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
        
        vals_acc = _get_acc_array(v, top_k=15)    
        ranks = [int(rank) for rank,_ in v.items()]
        means = vals_acc.mean(1)
        stds = vals_acc.std(1) / np.sqrt(vals_acc.shape[1])
        p_vals = []
        for i in range(vals_acc.shape[0] - 1):
            _, p = ttest_rel(vals_acc[i, :], vals_acc[i+1, :])
            p_vals.append(p)
        
        paired_sorted = sorted(zip(ranks, means, stds, p_vals))
        ranks, means, stds, p_vals = zip(*paired_sorted)
        bars = bars = plt.bar(ranks, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7, error_kw={'capthick': 2}, label='Mean ± SE')

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

def smooth_data(data, alpha=0.6):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Start with the first data point
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    
    return smoothed
    

def plot_accuracies_from_dicts(acc_dictionaries, top_k, save_name, skips=[], extras=[], lims=None):
    
    plt.figure(figsize=(10, 5))
    # mpl.style.use('seaborn-whitegrid')
    # Font settings
    plt.rc('font', family='Arial', size=12)
    
    max_rank = 0
    #read accuracies for all layers, all ranks
    blues = plt.cm.Blues(np.linspace(0.4, 1, 8))
    greens = plt.cm.Greens(np.linspace(0.4, 1, 8))
    c_idx = 0
    for k,v in acc_dictionaries.items():
        vals_acc = []
        ranks = []
        for r,v_acc in v.items():
            if str(r) in skips:
                continue
            v_acc= sorted(v_acc)[-top_k:]
            vals_acc.append([acc* 100 for acc in v_acc])
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
        if c_idx == 0:
            plt.errorbar(ranks, means, yerr=stds, label=f'{k}: Mean ± STD', fmt='-o', capsize=5, color=blues[4])
        else:
            plt.errorbar(ranks, means, yerr=stds, label=f'{k}: Mean ± STD', fmt='-o', capsize=5, color=greens[4])
        c_idx+=1
    
    for dictionary, name in extras:
        vals_acc =[]
        for _,v_acc in dictionary.items():
            v_acc= sorted(v_acc)[-top_k:]
            vals_acc.append(v_acc)
            vals_acc = np.array(vals_acc) * 100
        vals_acc = np.array(vals_acc)
        mean = vals_acc.mean(1)
        
        plt.plot([2,max_rank], [mean, mean], linestyle='-.', linewidth=2, color='black')
        std = vals_acc.std(1)
        
        plt.errorbar([16], [mean],yerr=[std], fmt='-.', capsize=5,
        capthick=2, ecolor='black', marker='s', 
        markersize=7, linestyle='-.', linewidth=2, label=f'{name}: Mean ± STD', color='black')
        
        
    # plt.grid(True)
    plt.xlabel('Backward Rank', fontsize=18, fontname='Arial')
    plt.ylabel('Accuracy', fontsize=18, fontname='Arial')
    # plt.title('Mean Accuracy and Standard Error over Trials at each Layer', fontsize=14, fontname='Arial')
    plt.xticks([1, 4, 8, 16, 32])
    plt.xticks(fontsize=15, fontname='Arial')
    plt.yticks(fontsize=15, fontname='Arial')
    if lims is not None:
        plt.ylim(lims)
    plt.legend(fontsize=18)
    plt.savefig(save_name)
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

def get_accuracies_from_training_path(path):
    
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert Images to Grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    
    batch_size = 64
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    path = Path(path)
    accuracies = dict()
    for rank_folder in path.iterdir():
        if rank_folder.stem == 'BP':
            rank = 'BP'
        else:
            continue
#        rank = int(rank_folder.stem.split('_')[-1])
        accuracies[rank] = []
        for exp_folder in tqdm(rank_folder.iterdir()):
           weights_path = exp_folder / 'checkpoints' / 'best.pth'
        #    model = AlexNet_AFA(1, kernel_size=9, bn = 32, num_classes=10, device='cuda', constrain=rank)
           model = AlexNet_cifar(1, kernel_size=9, bn = 32, num_classes=10, device='cuda')
           model.load_state_dict(torch.load(weights_path))
           model.to('cuda')
           accuracy = evaluate_model(model, testloader)
           accuracies[rank].append(accuracy)
    with open(path / 'accuracies.json', 'w') as f:
        json.dump(accuracies, f)
        

def generate_receptive_fields(path, model, layer_idx):
    
    path = Path(path)
    for rank_folder in path.iterdir():
        
        if not rank_folder.is_dir():
            continue
        rank = int(rank_folder.stem.split('_')[-1])
        for exp_folder in tqdm(list(rank_folder.iterdir())[::-1]):
           weights_path = exp_folder / 'checkpoints' / 'best.pth'
           receptive_fields_path = exp_folder / 'retina_rfs'
           receptive_fields_path.mkdir(exist_ok=True)
        #    model.vvs[0] = AFAConv2d(32, 32, 9, rank=rank, padding=9//2)
           model.load_state_dict(torch.load(weights_path), strict=False)
           model.eval()
           model.to('cuda')
           retina = model.retina
           montage = get_rfs(retina)
           plt.imsave(receptive_fields_path / 'retina_rfs.png', montage, cmap='gray')
           

def get_rfs(retina):
    
    filters = []
    for i in range(32):
        img = torch.zeros((1,1, 32, 32), requires_grad=True, device='cuda')
        sgd_optimizer = optim.SGD([img], lr=1, momentum=0)
        for _ in range(2):
            sgd_optimizer.zero_grad()
            output = retina(img)
            l = -1 * output[:,i, 16, 16]
            l.backward()
            sgd_optimizer.step()
    
        # img = (img - img.mean()) / (img.std() + 1e-5).sqrt()
        filters.append((img.detach().cpu().numpy()[0,0], l.item()))

    filters = sorted(filters, key = lambda x: x[1])
    # filters_vals = np.array([v[1] for v in filters])
    # filters_vals_mean = filters_vals.mean()
    filters_imgs = [(v[0] - v[0].mean()) / (v[0].std() + 1e-5) for v in filters]# if v[1] > filters_vals_mean - 0.5*filters_vals.std()]
    montage = create_image_montage(filters_imgs)
    return montage

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



def updateQ_vs_updateQP():
    
    P_acc_dict = load_dict_from_json(rf"artifacts/cifar10/constrain_all_update_Q_only/accuracies.json")
    QP_acc_dict = load_dict_from_json(rf"artifacts/cifar10/constrain_all_update_QP/accuracies.json")


def remove_epochs(path):
    
    path = Path(path)
    for rank_folder in path.iterdir():
        if rank_folder.is_dir():
            for exp_folder in tqdm(rank_folder.iterdir()):
                weights_path = exp_folder / 'checkpoints'
                for file in weights_path.iterdir():
                    if 'final' not in file.name:
                        file.unlink()
            
           
           
if __name__ == "__main__":
    
    
    # updateQ_vs_updateQP()
        
    # get_accuracies_from_training_path(rf"artifacts/cifar10/constraint all/")
    
    from models.layers.kDPFA import RConv2d as rADFAConv
    model = AlexNet_cifar(1, kernel_size=9, bn = 32, num_classes=10, device='cuda')
    # model.vvs[0] = AFAConv2d(32, 32, kernel_size=9, rank=32, padding=9//2)
    generate_receptive_fields(r'artifacts/cifar10/Retina_model_constrain_retina_only_lr_5e4_wd_5e5_gamma_975/vvs1', model, layer_idx=3)


    # # remove_epochs(r'/home/maherhanut/Documents/projects/EarlyVisualRepresentation_pfa/artifacts/cifar/bn_32_2vss_layers/vvs_2')
    
    