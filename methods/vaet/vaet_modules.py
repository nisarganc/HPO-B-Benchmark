'''
A Generative HPO method to observe and suggest new hyperparameter configurations. 
For each trial, this class can be used to train VAET model to suggest new hyperparameter configurations.
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
import os
import time
import copy
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from vaet_model import VAET_Model
from vaet_utils import TriplesDataset, totorch, collate_fn, device

class generativeHPO(nn.Module):
    def __init__(self, search_space_id, dataset_id, seed, params, torch_seed, verbose = False):
        super(generativeHPO, self).__init__()
        print("Using generativeHPO Process as method...") 
        torch.manual_seed(torch_seed)
        
        self.params = params
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id
        self.seed = seed
        self.verbose = verbose

        self.device = device()
        print("Using device: ", self.device)
        self.dataset = TriplesDataset(self.params, 'train')
        self.first_history = True

        rootdir = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(rootdir, "model_checkpoints/")
        os.makedirs(self.path, exist_ok=True)

    def get_model(self):
        model = VAET_Model(self.params) 
        self.model = model.to(self.device)

    def loss(self, predict):
        res = predict[0]

        x = predict[1]
        mu = predict[2]
        var = predict[3]

        # without mean over batch
        # loss1 = F.mse_loss(res, x, reduction='sum')
        # loss2 = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

        # Reconstruction loss
        loss1 = F.mse_loss(res, x, reduction='mean')

        # KL divergence loss
        loss2 = torch.mean(-0.5 * torch.sum((1 + var - mu**2 - torch.exp(var)),dim=1), dim=0)
        loss2.required_grad = True
        return loss1 + loss2 

    def train_and_eval(self, dataset):
        
        # load the model
        if os.path.exists(self.path+f"{self.search_space_id}_{self.dataset_id}_{self.seed}.pt"):
            self.model.load_state_dict(torch.load(self.path+f"{self.search_space_id}_{self.dataset_id}_{self.seed}.pt"))

        self.model.train()
        train_loader = DataLoader(dataset, batch_size=self.params['batch_size'],
                                   collate_fn=collate_fn, drop_last=True, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        best_loss = float('inf')
        epoch = 0
        
        while epoch < self.params['max_epochs']:  
            total_loss = 0
            for sample in train_loader:
                x = sample['x']
                I = sample['I']
                C = sample['C']
                mask = sample['mask']
                x, I, C, mask = x.to(self.device), I.to(self.device), C.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                predict = self.model(x, I, C, mask)
                loss = self.loss(predict)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            epoch += 1
            epoch_loss = total_loss/len(train_loader)

            # Save the model if it has the best evaluation loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.path+f"{self.search_space_id}_{self.dataset_id}_{self.seed}.pt")
                if best_loss < 0.1:
                    break

            if self.verbose:
                print(f"Epoch {epoch}, loss: {epoch_loss}")
                print(f"Best loss: {best_loss}")
        
        
        # load the best model
        self.model.load_state_dict(torch.load(self.path+f"{self.search_space_id}_{self.dataset_id}_{self.seed}.pt"))
        self.model.eval()
        dataset.load_eval_data()
        eval_loader = DataLoader(dataset, batch_size=1, 
                                 collate_fn=collate_fn, drop_last=True, shuffle=False)
        with torch.no_grad():
            for sample in eval_loader:
                I = sample['I']
                C = sample['C']
                mask = sample['mask']
                I, C, mask = I.to(self.device), C.to(self.device), mask.to(self.device)
                xnew = self.model.generate(I, C, mask)
                
        print(f"Epochs: {epoch} Best loss: {best_loss}")        
        return xnew.squeeze(0)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        
        # Get sampled Triples dataset (x, I, C) 
        if self.first_history:
            self.params['obs'] = len(X_obs)
            self.dataset.load_history(totorch(X_obs, self.device), totorch(y_obs, self.device).reshape(-1))
            self.get_model()
            self.first_history = False
        else:
            self.dataset.update_history(totorch(X_obs[-1], self.device), totorch(y_obs[-1], self.device).reshape(-1))

        # Train the model 
        x_new = self.train_and_eval(self.dataset)
        # print("X_new: ", x_new)

        return x_new.cpu().numpy()
    
class VAET(nn.Module):
    def __init__(self, train_data, valid_data, checkpoint_path, args ):
        super(VAET, self).__init__()
        self.params = args
        self.checkpoint_path = checkpoint_path

        self.train_data = train_data
        self.valid_data = valid_data
        first_dataset = list(self.train_data.keys())
        print("Train data: ", first_dataset)
        self.params.input_size = len(train_data[first_dataset]["X"][0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training params
        self.lr = args.lr
        self.epochs = args.max_epochs
        
        exit()
       