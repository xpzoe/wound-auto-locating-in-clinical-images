import pickle
import os 
import seaborn as sns
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
import numpy as np
from torchvision.io import read_image
from PIL import Image
from torchvision import models
import ast

plt.ion()

def load_results(folder_path):
    '''
    Load results.

    Args:
        folder_path: string, path to each test result folder
    
    Return: 
        model_type, lr, acc, df, pred_dict
    '''

    folder_name = os.path.basename(folder_path)
    model_type = folder_name.split('_') # split result folder name by '_'

    # find model_type, always at the beginning of the folder name
    if len(model_type)==8:
        model_type = model_type[0]
    elif len(model_type)==9:
        model_type = model_type[0]+'_'+model_type[1]
    elif len(model_type)==10:
        model_type = model_type[0]+'_'+model_type[1]+'_'+model_type[2]

    for f in os.listdir(folder_path): 
        # find text file, get lr and so on
        if 'txt' in f: 
            with open(os.path.join(folder_path, f), 'r') as handle:
                config = handle.readlines()
                lr = float(config[1].split()[1])
                if 'OneMoreFC' in model_type:
                    fc_size = config[-1].split()[-1]
                elif 'twoMoreFC' in model_type:
                    fc_size = '1000+256'
                else:
                    fc_size = 'none'

        # find prediction file
        elif 'preds' in f:  
            with open(os.path.join(folder_path, f), 'rb') as handle:
                preds_dict = pickle.load(handle)
                acc = preds_dict.pop('avg_acc')
                loss = preds_dict.pop('avg_loss')

        elif 'xlsx' in f:
            df = pd.read_excel(os.path.join(folder_path, f))
        else:
            continue

    return model_type, lr, acc, df, preds_dict, fc_size

def save_lr_acc_model(parent_folder_path, save_folder):
    '''
    Save lr_acc_models to excel.
    
    Args:
        parent_folder_path: string, the parent folder where all results folders locate
        save_folder: string, the folder to save dataframe
    '''
    save_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = 'lr_acc_model'
    df = pd.DataFrame(columns=['model type', 'learning rate', 'accuracy'])
    for f in os.listdir(parent_folder_path):
        model, lr, acc, _, _= load_results(os.path.join(parent_folder_path, f))
        df = df.append({'model type': model, 'learning rate': lr, 'accuracy': acc}, ignore_index=True)
    df.to_excel(os.path.join(save_folder, save_name+'_'+save_time+'.xlsx'), index=False)

def save_fc_acc_model(parent_folder_path, save_folder):
    '''
    Save fc_acc_models to excel
    '''
    save_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = 'fc_acc_model'
    df = pd.DataFrame(columns=['model type', 'learning rate', 'accuracy'])
    for f in os.listdir(parent_folder_path):
        model, lr, acc, _, _, fc= load_results(os.path.join(parent_folder_path, f))
        df = df.append({'model type': model, 'learning rate': lr, 'accuracy': acc, 'fc': fc}, ignore_index=True)
    df.to_excel(os.path.join(save_folder, save_name+'_'+save_time+'.xlsx'), index=False)

def draw_pred_vs_label_save(parent_folder_path):
    '''
    Choose several best tests, draw pred vs label.

    Args:
        parent_folder_path: string, the parent folder where all results folders locate
    '''
    labels_map = {
        'cephalon': 0,
        'ear': 1,
        'shoulder': 2,
        'dorsum': 3,
        'elbow': 4,
        'lumbus': 5,
        'munus': 6,
        'knee': 7,
        'digits': 8,
        'calcaneus': 9,
        'planta':10,
        'others':11
    }
     
    wrong_pred_matrix = np.zeros([12, 12])
    pred_matrix = np.zeros([12, 12])
    for f in os.listdir(parent_folder_path):
        if 'test_on_real' in f:
            model_type, learning_rate, _, _, preds_dict, _ = load_results(os.path.join(parent_folder_path, f))
            # if model_type==model and learning_rate==lr:
            for key, value in preds_dict.items():
                label = key.split('/')[0]
                # pred = list(labels_map.keys())[value]
                label_int = labels_map[label]
                pred_matrix[int(value)][label_int] += 1
                # if int(value)!=label_int: wrong_pred_matrix[int(value)][label_int] += 1
                wrong_pred_matrix[int(value)][label_int] += 1
    sums = np.sum(wrong_pred_matrix, axis=0)
    pred_prob_matrix = np.zeros_like(wrong_pred_matrix)
    for i, sum in enumerate(sums):
        if sum != 0: 
            pred_prob_matrix[:,i] = wrong_pred_matrix[:,i]/sum 

    indices = list(labels_map.keys())
    df = pd.DataFrame(pred_matrix, index=indices, columns=indices)
    df_prob = pd.DataFrame(pred_prob_matrix, index=indices, columns=indices)
    df.head()
    df_prob.head()

    fig, axs = plt.subplots(1, 1, figsize=(15, 15))

    #ax = sns.heatmap(df, cmap='YlGn', annot=False, ax=axs[0])
    ax_prob = sns.heatmap(df_prob, cmap='YlGn', annot=True, ax=axs)

    #ax.set_xlabel('label')
    #ax.set_ylabel('prediction')
    ax_prob.set_xlabel('label')
    ax_prob.set_ylabel('prediction')
    #axs[0].set_title("Counts")
    #axs[1].set_title("Probabilities")
    plt.tight_layout()
    plt.show()

def show_aug(img_path):
    image = Image.open(img_path)
  
    v_flip = transforms.RandomVerticalFlip(p=1)
    h_flip = transforms.RandomHorizontalFlip(p=1)
    affine = transforms.RandomAffine(degrees=0, translate=[0.1, 0.2], scale=(0.75, 1), shear=15, fill=0)
    rot = transforms.RandomRotation(180)
    pers = transforms.RandomPerspective(distortion_scale=0.5, p=1)
    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
    sharp = transforms.RandomAdjustSharpness(2)

    t = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(p=1)
            ]),
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=0, translate=[0.1, 0.2], scale=(0.75, 1), shear=15, fill=0),
                transforms.RandomPerspective(distortion_scale=0.5, p=1),
            ]),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.RandomAdjustSharpness(2),
                transforms.RandomAutocontrast()
            ])
        ])

    image = image.convert('RGB')
    fig1, axs1 = plt.subplots(2, 3, figsize=(10, 6))
    axs1[0][0].imshow(image)
    axs1[0][1].imshow(t(image))
    axs1[0][2].imshow(t(image))
    axs1[1][0].imshow(t(image))
    axs1[1][1].imshow(t(image))
    axs1[1][2].imshow(t(image))
    axs1[0][0].set_title('Original Image')
    plt.tight_layout()


