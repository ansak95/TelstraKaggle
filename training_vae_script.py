import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm

import pickle
import pandas as pd
import datetime 

import matplotlib.pyplot as plt
from IPython.display import display

import os
import argparse
import random

import tqdm
import numpy as np
from VAE import VAE



def load_checkpoint(filepath, train = False):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    
    if train :
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()
    else :
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model


def training_args():
    parser=argparse.ArgumentParser(description='fine_tune')

    parser.add_argument('--path', default='', type=str,
                        help='model path')
    
    parser.add_argument('--device', default=0, type=int,
                        help='which device')
    
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of training epochs')
    
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='Learning rate')

    
    parser.add_argument('--wgt', default=1e-4, type=float,
                        help='Weight KL divergence')
    
    args=parser.parse_args()
    return args

def create_loaders(data, bs=512, jobs=0):
    data = DataLoader(data, bs, shuffle=True, num_workers=jobs, pin_memory = False)
    return data

# constants
args = training_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bs = args.batch_size
epochs = args.epochs
wgt = args.wgt


 
import os
# os.chdir("/home/anassakrim/FolderThesis/DB1602/FolderSSL_Nov21/Fine_tuning/TempHiddenState")
# fd = 'GenerateSet/'
# os.makedirs(fd)
# fd_km = fd + args.folder_data



# instantiate model
torch.manual_seed(7)
torch.cuda.manual_seed(7)


def create_loaders(data, bs=512, jobs=0):
    data = DataLoader(data, bs, shuffle=True, num_workers=jobs, pin_memory = False)
    return data
bs = 32

X_trn = torch.load('X_train.pt')
X_val = torch.load('X_val.pt')

trn_dl = TensorDataset(X_trn)#, Y_tr_torch)
trn_dl = create_loaders(trn_dl, bs, jobs=1)

val_dl = TensorDataset(X_val)#, Y_val_torch)
val_dl = create_loaders(val_dl, bs, jobs=1)


PATH = "model.pth"
fd = ""#args.folder
dt = f"{datetime.datetime.now():%Y%h%d_%Hh%M}" #datetime
path_model = fd + "VAE_extract_" + dt 
os.makedirs(path_model)
dir_path = path_model + "/"
import time



def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


nb_features = X_trn.shape[1]
lr = args.lr
it = 0
criterion = nn.MSELoss()
model = VAE(input_dim = nb_features, criterion = criterion, weight_kl = wgt)
nb_params = model.number_of_parameters()
print(nb_params)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-6)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

path_lr = dir_path + model.__class__.__name__ + f'_lr{lr}'
os.makedirs(path_lr)
dir_path = path_lr + "/"


#save the model architecture
f = open(dir_path+"model_parameters.txt", "a")
f.write(str(model.state_dict))
f.close()

# #save the log 
f = open(dir_path+"log_loss.txt", "a")
# f.write(str(model.state_dict))
f.close()


#save the args
f = open(dir_path+"args.txt", "w+")
f.write(str(args))
f.close()

#which optimizer
f = open(dir_path+"optim.txt", "w+")
f.write(str(optimizer))
f.close()                    

t0 = time.time()

#reset params
patience = 500
best_mape = 10000
trn_mape_track = []
val_mape_track = []




from tqdm import trange
it_lr = 0 #number of runs
for l_r in [args.lr/(10**p) for p in range(3)] :
    patience = 500

    if it_lr != 0 :
        print("Load the model...")
        f = open(dir_path+"log_loss.txt", "a")
        f.write("Load the model...")
        f.write("\n")
        f.close()

        model = load_checkpoint(dir_path+PATH, train = True)   
        model.to(device)
        checkpoint = torch.load(dir_path+PATH)
        optimizer = optim.Adam(model.parameters(), lr=l_r,)#, betas=(0.9, 0.95), eps=1e-08)
        optimizer.load_state_dict(checkpoint['optimizer_dic'])
        update_lr(optimizer, l_r)

        it_lr = it_lr+1



    # TRAINING    
    print('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
    f = open(dir_path+"log_loss.txt", "a")
    f.write("Begin training." + "\n")
    f.write('Learning rate adjusted to {:0.7f}'.format(optimizer.param_groups[0]['lr']))
    f.write("\n")
    f.close()

    pbar = trange(args.epochs,  unit="epoch")
    for epoch in pbar:
        time.sleep(0.1)
        model.train()
        t1 = time.time()
        model.train_model(trn_dl,optimizer)


        model.eval()
        with torch.no_grad() :

            train_mape = model.eval_mape(trn_dl)
            trn_mape_track.append(train_mape)

            val_mape = model.eval_mape(trn_dl)
            val_mape_track.append(val_mape)



        pbar.set_description(f'Epoch {epoch+1}/{args.epochs}')
        pbar.set_postfix_str(f'Train set mape {train_mape:2.2%}, Val set mape {val_mape:2.2%}, Best mape {best_mape:2.2%}, Patience = {patience}')
        f = open(dir_path+"log_loss.txt", "a")
        f.write(f'Train set mape {train_mape:2.2%}, Val set mape {val_mape:2.2%}, Best mape {best_mape:2.2%}, Patience = {patience}')
        f.write("\n")
        f.close()



        if epoch%100 == 0 :   
            torch.cuda.empty_cache() 
        if val_mape < best_mape :
            epoch_save = epoch
            patience = 500
            best_mape = val_mape
            f = open(dir_path+"log_loss.txt", "a")
            f.write("Save the model...")
            f.write("\n")
            f.close()

            checkpoint = {'model': model, 'mape': trn_mape_track, 'val_mape': val_mape_track,  
                  'state_dict': model.state_dict(),
                  'optimizer_dic' : optimizer.state_dict(), 'lr' : lr}
            torch.save(checkpoint, dir_path+PATH)
        else : 
            patience = patience - 1

        if patience == 0 :
            pbar.set_postfix_str(f'Ended, Best mape {best_mape:2.2%}')
            break


plt.rc('figure',figsize=(22,12))
plt.rcParams.update({'font.size': 22})
plt.rc('ytick', labelsize=18)
plt.rc('xtick', labelsize=18)


plt.plot(trn_mape_track, label =  f'Training set')
plt.plot(val_mape_track, label =  f'Val set with best mape = {best_mape:2.2%}')#.cpu().numpy()

# plt.ylim(0,20)
plt.grid()
plt.legend()
plt.title(f'Features extraction with VAE')

plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')   
plt.savefig(dir_path+f'Fold{it}.jpg')
plt.close()


#del model and empty cache
del(model)
torch.cuda.empty_cache()  



    




























        
