import time
import copy
import torch
from torch.autograd import Variable
from tqdm import trange
import tqdm
import os
import pandas as pd

def train_model(dataloaders, dataset_sizes, device, model, criterion, optimizer, scheduler, num_epochs, results_folder, network_name):
    '''
    Function which executes training. 
    
    Args:
        dataloaders: DataLoader dict
        dataset_sizes: dict, sizes of subsets
        device: 'cuda' or 'cpu'
        model: string, which model to train
        criterion, optimizer: loss function and optimizer
        num_epochs: int
        results folder: string, folder to save losses
    
    Returns: 
        model: model after training 
        best_epoch: the epoch which gives out the best validation accuracy
    '''

    network_list = ['VGG16_OneMoreFC_', 'RESNET50_OneMoreFC_', 'VGG19_OneMoreFC_']
    with_val = True
    if device=='cuda:0': print("[INFO] Using CUDA")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    df = pd.DataFrame(columns=['Epoch', 'Loss'])
    
    # start training
    for epoch in range(num_epochs):
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)

        # process bar
        loop = tqdm.tqdm((dataloaders['train']), total = len(dataloaders['train']))
        current_number = 0

         # compute output and loss for each sample batch, back propagation
        for inputs, labels, names in loop:  

            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            optimizer.zero_grad()

            if network_name in network_list:
                outputs = model(inputs) # output is probability array, (12,4)
                _, preds = torch.max(outputs.data, 1) # prediction is where the largest probability locates
                loss = criterion(outputs, labels.float()) # compute loss
            else: # special for inception v3
                outputs, aux_outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss1 = criterion(aux_outputs, labels.float())
                loss2 = criterion(outputs, labels.float())
                loss = loss1*0.4 + loss2

            loss.backward()
            optimizer.step()
            
            try:
                loss_train += loss.data[0]
            except:
                loss_train += loss.data
            
            # transfer one-hot back to int
            label_int = labels.data.tolist()
            for k,l in enumerate(label_int):
                label_int[k] = l.index(1)

            # count when prediction equals to label
            acc_train += torch.sum(preds == torch.tensor(label_int).to(device))
            
            del inputs, labels, outputs, preds, label_int
            torch.cuda.empty_cache()

            # process bar
            current_number += len(inputs)
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(train_loss = loss_train / current_number, train_acc = acc_train / current_number)

        avg_loss = loss_train / dataset_sizes['train']
        avg_acc = acc_train / dataset_sizes['train']

        # validation, pretty much similar to training, but without computing grads
        if with_val:
            model.train(False)
            model.eval()
        
            val_loop = tqdm.tqdm((dataloaders['val']), total = len(dataloaders['val']))
            val_current_number = 0

            for inputs, labels, names in val_loop:
                    

                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.float())
                
                try:
                    loss_val += loss.data[0]
                except:
                    loss_val += loss.data
                
                label_int = labels.data.tolist()
                for i,l in enumerate(label_int):
                    label_int[i] = l.index(1)
                acc_val += torch.sum(preds == torch.tensor(label_int).to(device))
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

                val_current_number += len(inputs)
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(val_loss = loss_val / val_current_number, val_acc = acc_val / val_current_number)

            avg_loss_val = loss_val / dataset_sizes['val']
            avg_acc_val = acc_val / dataset_sizes['val']
        df = df.append({'Epoch': epoch+1, 'train Loss': avg_loss.item(), 'train acc': avg_acc.item(),'val Loss': avg_loss_val.item(), 'val acc': avg_acc_val.item()}, ignore_index=True)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_epoch = epoch
    
    # save loss and acc during training and val
    df.to_excel(os.path.join(results_folder, 'training_loss.xlsx'), index=False)

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc), "Best epoch: {:d}".format(best_epoch))
    
    return model, best_epoch
