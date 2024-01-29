import time
from torch.autograd import Variable
import torch
from tqdm import tqdm
from tqdm import trange
import os
from datetime import datetime
import pickle

def eval_model(dataloaders, dataset_sizes, device, model, criterion, network_name):
    '''
    Using a trained model to predict on test data.

    Agrs:
        dataloaders: DataLoader dict
        dataset_sizes: dict, sizes of subsets
        device: 'cuda' or 'cpu'
        model: trained model
        criterion: loss function
        network_name: string
    Return: 
        preds_dict: predictions arrays for each test sample
        prob_dict: probability arrays for each test sample
    '''

    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders['test'])
    print("Evaluating model")
    print('-' * 10)

    if network_name == 'INCEPTIONV3_':
        model.aux_logits = False

    preds_dict = {}
    prob_dict = {}
    
    for k, data in enumerate(dataloaders['test']):

        model.train(False)
        model.eval()
        
        inputs, labels, names = data
        inputs, labels = Variable(inputs.to(device), volatile=True), Variable(labels.to(device), volatile=True)

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels.float())

        for i, name in enumerate(names):
            preds_dict[name] = preds[i].cpu().numpy()
            prob_dict[name] = outputs[i].cpu().detach().numpy()

        try:
            loss_test += loss.data[0]
        except:
            loss_test += loss.data
        
        label_int = labels.data.tolist()
        for i,l in enumerate(label_int):
            label_int[i] = l.index(1)
        acc_test += torch.sum(preds == torch.tensor(label_int).to(device))

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / dataset_sizes['test']
    avg_acc = acc_test / dataset_sizes['test']

    preds_dict['avg_loss'] = avg_loss.cpu().numpy()
    preds_dict['avg_acc'] = avg_acc.cpu().numpy()
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)

    return preds_dict, prob_dict
