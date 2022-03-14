import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable
from torch.nn.modules import ModuleList
import copy

import numpy as np
import os
from tqdm import tqdm_notebook, trange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#no dropout since the vae is already a bayesian model
class VAE(nn.Module):
    def __init__(self,  input_dim, criterion, weight_kl = 1e-4):
        super(VAE, self).__init__()
        
        self.w = weight_kl
        self.encoder = nn.Sequential(nn.Linear(input_dim,128),
              nn.LayerNorm(128, elementwise_affine=False),                    
              nn.GELU(),
                                     
              nn.Linear(128,64),
              nn.LayerNorm(64, elementwise_affine=False),                       
              nn.GELU()).to(device)
        
        self.fc_enc_mu = nn.Sequential(nn.Linear(64,32),
              nn.LayerNorm(32, elementwise_affine=False),                      
              nn.GELU()).to(device)
        
        self.fc_enc_logvar = nn.Sequential(nn.Linear(64,32),
              nn.LayerNorm(32, elementwise_affine=False),                      
              nn.GELU()).to(device)
        
        self.decoder = nn.Sequential(nn.Linear(32,64),
              nn.LayerNorm(64, elementwise_affine=False),                    
              nn.GELU(),
                                     
              nn.Linear(64,128),
              nn.LayerNorm(128, elementwise_affine=False),
              nn.GELU()).to(device)
        
        
        self.ln = nn.Linear(input_dim,64).to(device)
        self.norm = nn.LayerNorm(64, elementwise_affine=False).to(device)

            
        self.fc = nn.Linear(128,input_dim).to(device)
        
        self.criterion = criterion.to(device)
        
        
#         self.mean = mean_val.to(device)
#         self.std = std_val.to(device)

        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)     
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)) :#and ():
            nn.init.xavier_normal_(module.weight.data)
                    
#     def normalize(self, input) :
#         return (input-self.mean)/self.std
#     def inv_norm(self, input) :
#         return input*self.std+self.mean
    
    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, input):

#         input = self.normalize(input).to(device)       
#         input = torch.nan_to_num(input, nan=0.0)
        
        #encode
        embedding = self.encoder(input) 
#         print(self.norm(self.ln(input)).shape)
#         embedding = embedding + self.norm(self.ln(input))
        # Split the result embedding into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_enc_mu(embedding)
        log_var = self.fc_enc_logvar(embedding)
        
        #compute the latent embedding
        z = self.reparameterize(mu, log_var)
        
        
        #decode
        if self.train :
            out = self.decoder(z) 
            out = self.fc(out)
#             print(out.shape)
            reconstruction_error = self.criterion(out,input)
            kl_divergence = (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))
            loss = (reconstruction_error + self.w*kl_divergence).sum()
            return out, z, mu, log_var, loss
        else :
            out = self.decoder(mu)
            out = self.fc(out)
            return out, z, mu, log_var
        
        
    def train_model(self, loader, optimizer) :
        loss = 0   
        for i, data in enumerate(loader):   
            X_train_batch = data[0].to(device)
            optimizer.zero_grad()
            loss = self.forward(X_train_batch)[-1] 
#             print(loss)
            loss.backward()
            optimizer.step()
        
    def eval_mape(self, loader) :
        metric_mape = 0
        with torch.no_grad() :
            for i, data in enumerate(loader):   
                x = data[0].to(device)
                x_out = self.forward(x)[0]
                mape = torch.abs((x_out-x)/x)
                mape[torch.isnan(mape)] = 0
                mape = mape.masked_fill(mape.isinf(),0)
                metric_mape += torch.mean(torch.abs(mape)).item()
        return metric_mape/(i+1)
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))









