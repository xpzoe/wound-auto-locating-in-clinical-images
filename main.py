import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from my_models_torch import my_vgg16, my_resnet50, my_inceptionv3,my_vgg19, my_vgg16_two, my_vgg19_two
from training import train_model
from testing import eval_model
from prepare_dataset_new import  load_data, load_data_testonreal
from datetime import datetime
import os
import pickle
import random
import numpy as np


def run(network_name, dataset_type, data_dir, load_weights, lr, num_epochs):
    '''
    Execute training or testing run, for each configuration.

    Args:
        network_name: string, name of classification model used
        dataset_type: string, name of dataset
        data_dir: string, path to dataset
        load_weights: string, pretrained big dataset
        lr: float, learning rate
        num_epochs: int
    '''

    # print(network_name)
    # print(lr)
    # print(num_epochs)

    use_gpu = torch.cuda.is_available()

    # create result folder for single run
    save_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(r'/path/to/result/folder', network_name+dataset_type+save_time)
    os.mkdir(results_folder)

    dataloaders, datasizes= load_data(data_dir, network_name)

    network_dict= {
        'RESNET50_OneMoreFC_': my_resnet50(load_weights),
        'VGG16_OneMoreFC_': my_vgg16(load_weights),
        'VGG19_OneMoreFC_': my_vgg19(load_weights)
    }
    network = network_dict.get(network_name)

    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # weighted loss in order: 
    # ['cephalon', 'ear', 'shoulder', 'dorsum', 'elbow', 'lumbus', 'munus', 'knee', 'digits', 'calcaneus', 'planta', 'others']
    weight = torch.from_numpy(np.array([2.11,1.06,1,1,0.94,0.96,0.94,0.88,1.59,0.78,2.23,1])) 
    criterion = nn.CrossEntropyLoss(weight=weight) 
    optimizer_ft = optim.SGD(network.parameters(), lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    print('[INFO] Start training...')
    model, best_epoch  = train_model(dataloaders, datasizes, device, network, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, results_folder, network_name)

    # save necassary configurations
    configs = ['criterion: '+str(criterion)+'\n', 
               'lr: '+str(optimizer_ft.defaults['lr'])+'\n', 
               'step_size: '+str(exp_lr_scheduler.step_size)+'\n',
               'gamma: '+str(exp_lr_scheduler.gamma)+'\n',
               'num_epochs: '+str(num_epochs)+'\n',
               'best_val_epoch: '+str(best_epoch)+'\n',
               'pre_trained: True'+'\n'
               # '---------------------------------',
               ]
    with open(os.path.join(results_folder, 'config.txt'), 'w') as handle:
        handle.writelines(configs)

    # save model
    save_name = network_name + load_weights + '.pth'
    torch.save(model.state_dict(), os.path.join(results_folder, save_name))

    testing = True
    if testing:
        print('[INFO] Start test...')
        preds_dict, probs_dict = eval_model(dataloaders, datasizes, device, model, criterion, network_name)
        acc = preds_dict['avg_acc']  
        # save test results    
        preds_save_path = os.path.join(results_folder, f'preds_and_acc{acc}.pkl')
        probs_save_path = os.path.join(results_folder, f'probs_and_acc{acc}.pkl')
        with open(preds_save_path, 'wb') as handle:
            pickle.dump(preds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(probs_save_path, 'wb') as handle2:
            pickle.dump(probs_dict, handle2, protocol=pickle.HIGHEST_PROTOCOL)

def main():
        
    data_dir = r'/path/to/dataset'
    dataset_type = 'my_images_'

    network_name = ['VGG16_OneMoreFC_', 'RESNET50_OneMoreFC_', 'VGG19_OneMoreFC_']
    load_weights = 'DEFAULT' # 'DEFAULT' refers to ImageNet-1k

    lr = [0.0005] 
    num_epochs = [50] 

    for t in range(10):
        run(random.choice(network_name), dataset_type, data_dir, load_weights, lr[0], num_epochs[0])


if __name__ == "__main__":
    main()






