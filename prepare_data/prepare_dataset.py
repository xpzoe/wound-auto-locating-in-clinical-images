import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
from torchvision.io import read_image
from torchvision import transforms
import pickle
import tqdm


def transformers_f(network_name):

    # policy = transforms.AutoAugmentPolicy.IMAGENET

    input_sizes = {
        'RESNET50_': [224, 224], 
        'VGG16_': [224, 224], 
        'INCEPTIONV3_': [299, 299]
    }

    data_transforms = {
        'default': transforms.Compose([
            transforms.Resize([256, 256]),
        ]),
    
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_sizes[network_name]), 
            # transforms.AutoAugment(policy),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=0, translate=[0.1, 0.2], scale=(0.75, 1), shear=15, fill=0),
                transforms.RandomRotation(180),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomHorizontalFlip(),
            ], 0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.RandomAdjustSharpness(2),
                transforms.RandomAutocontrast(),
            ], 0.7),
            # transforms.ToTensor(),
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

def create_annotation_file(data_dir):
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
    annotation_file_name = 'annotation.pkl'
    annotation_dict = {}
    class_lists = os.listdir(data_dir)
    if annotation_file_name in class_lists: 
        return os.path.join(data_dir, annotation_file_name)
    one_hot = torch.nn.functional.one_hot(torch.tensor(list(labels_map.values())), num_classes=len(labels_map)+1)
    for nclass in class_lists:
        for image in os.listdir(os.path.join(data_dir, nclass)):
            annotation_dict[os.path.join(nclass,image)] = one_hot[labels_map[nclass]][0:12]
    with open(os.path.join(data_dir, annotation_file_name), 'wb') as handle:
        pickle.dump(annotation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.join(data_dir, annotation_file_name)


class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
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
        if image.shape[0] == 4: image = image[0:3, :, :]
        label = img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_names[idx]


def load_data(data_dir, network_name):
    annotation_file = create_annotation_file(data_dir)
    transformers = transformers_f(network_name)
    dataset = MyDataset(annotation_file, data_dir, transform=transformers['default'], target_transform=None)

    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2

    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 4
    shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

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
