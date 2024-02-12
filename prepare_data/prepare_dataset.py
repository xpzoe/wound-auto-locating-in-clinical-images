import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from torchvision.io import read_image
from torchvision import transforms
import pickle
import tqdm
import numpy as np


def transformers_f(network_name):
    '''
    Augmentation, return a transformer dict.
    
    Args: 
        network_name: the string of model name, used for deciding input size
        
    Return: 
        data_transform: dictionary of transformers, each item is a torch transform object
    '''

    # specify input size for different models 
    input_sizes = {
        'RESNET50_OneMoreFC_': [224, 224], 
        'VGG16_OneMoreFC_': [224, 224], 
        'INCEPTIONV3_OneMoreFC_': [299, 299],
        'VGG19_OneMoreFC_': [224,224],
        'VGG16_twoMoreFC_': [224,224],
        'VGG19_twoMoreFC_': [224,224]
    }

    data_transforms = {
        'default': transforms.Compose([
            transforms.Resize(input_sizes[network_name]),
        ]), # resize to the defined input size for train, val and test sets

        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_sizes[network_name]), 

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
            ]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),


        'val': transforms.Compose([
                    transforms.CenterCrop(input_sizes[network_name]),
                    # transforms.ToTensor(),
                ]),

        'test': transforms.Compose([
            transforms.CenterCrop(input_sizes[network_name]),
            # transforms.ToTensor(),
        ])
    }
    return data_transforms

def create_annotation_file(data_dir, annotation_file_name):
    '''
    Automatically create annotation file for images in the given folder. 
    Images should be located in a subfolder named by its class, the annotation format is {'class/random image name': one-hot label}
    
    Args: 
        data_dir: string, path to dataset folder
        annotation_file_name: string, the name of annotation file
        
    Return: absolute path to annotation file
    '''

    # give each class an int-label
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
    
    annotation_dict = {}
    class_lists = os.listdir(data_dir)
    if annotation_file_name in class_lists: 
        return os.path.join(data_dir, annotation_file_name)
    
    # transfer int-label to one-hot
    one_hot = torch.nn.functional.one_hot(torch.tensor(list(labels_map.values())), num_classes=len(labels_map)+1)
    
    # create dict
    for nclass in class_lists:
        for image in os.listdir(os.path.join(data_dir, nclass)):
            annotation_dict[os.path.join(nclass,image)] = one_hot[labels_map[nclass]][0:12]

    with open(os.path.join(data_dir, annotation_file_name), 'wb') as handle:
        pickle.dump(annotation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.join(data_dir, annotation_file_name)

class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        '''
        Args:
            annotation_file: string, absolute path to annotation file
            img_dir: string, the path to dataset folder
            transform: image transform
            target_transform: label transform
            
        Return: image, label and image name for each sample
        '''
        with open(annotations_file, 'rb') as handle:
            self.img_labels = pickle.load(handle)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_names = list(self.img_labels.keys())
        img_labels = list(self.img_labels.values())
        img_path = os.path.join(self.img_dir, img_names[idx])
        image = read_image(img_path).float()
        if image.shape[0] == 4: image = image[0:3, :, :] # only keep rgb channel when encountering 4-channel images  
        label = img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_names[idx]


def load_data(data_dir, network_name):
    '''
    Load images, create annotation file and create the DataLoader object for each subset.
    
    Args: 
        data_dir: path to data folder
        network_name: model name, pass to resize images
        
    Returns:
        loaders: dict, the DataLoader object for each subset
        sizes: dict, sizes of each subset 
    '''

    print('[INFO] Preparing dataset...')

    annotation_file = create_annotation_file(data_dir, 'annotation.pkl')
    transformers = transformers_f(network_name)

    dataset = MyDataset(annotation_file, data_dir, transform=transformers['default'], target_transform=None)

    # randomly splite dataset to 3 subsets
    train_ratio = 1.0
    val_ratio = 0.2
    test_ratio = 0.3

    train_size = int(len(dataset) * train_ratio)
    # val_size = int(len(dataset) * val_ratio)
    # test_size = int(len(dataset) * test_ratio)

    np.random.seed(random_seed)
    
    train_indices = np.random.choice(dataset_size, size=train_size, replace=True)

    remaining_indices = np.setdiff1d(np.arange(dataset_size), train_indices)
    test_size = int(len(remaining_indices) * test_ratio / (1 - test_ratio - validation_ratio))
    test_indices = np.random.choice(remaining_indices, size=test_size, replace=False)

    validation_indices = np.setdiff1d(remaining_indices, test_indices)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # shuffle = True
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # apply augmentation transforms 
    for images, _, _ in train_loader:
        images = transformers['train'](images)

    for images, _, _ in val_loader:
        images = transformers['val'](images)

    for images, _, _ in test_loader:
        images = transformers['test'](images)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    sizes = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }

    return loaders, sizes
