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
from vaet_model import VAET
from vaet_utils import TriplesDataset, totorch, collate_fn

class generativeHPO(nn.Module):
    def __init__(self, params, seed, verbose = False):
        super(generativeHPO, self).__init__()
        print("Using generativeHPO Process as method...") 
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params['device'] = self.device
        self.params = params
        self.verbose = verbose

    def get_model(self):
        model = VAET(self.params) 
        self.model = model.to(self.device)

    def loss(self, predict):
        res = predict[0]
        x = predict[1]
        mu = predict[2]
        var = predict[3]

        # Reconstruction loss
        loss1 = F.mse_loss(res, x, reduction='sum')

        # KL divergence loss
        loss2 = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

        return loss1 + loss2 

    def train_and_eval(self, train_dataset, eval_dataset, save_path):

        self.model.train()
        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'],
                                   collate_fn=collate_fn, drop_last=True, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])

        best_loss = float('inf')
        for epoch in range(self.params['epochs']):
            
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

            epoch_loss = total_loss/len(train_loader)

            # Save the model if it has the best evaluation loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), save_path+"best_model.pt")
                if best_loss < 0.001:
                    break

            if self.verbose:
                print(f"Epoch {epoch}, loss: {epoch_loss}")
                print(f"Best loss: {best_loss}")
        
        
        # load the best model
        self.model.load_state_dict(torch.load(save_path+"best_model.pt"))
        self.model.eval()
        eval_loader = DataLoader(eval_dataset, batch_size=1, 
                                 collate_fn=collate_fn, drop_last=True, shuffle=False)
        with torch.no_grad():
            for sample in eval_loader:
                I = sample['I']
                C = sample['C']
                mask = sample['mask']
                I, C, mask = I.to(self.device), C.to(self.device), mask.to(self.device)
                xnew = self.model.generate(I, C, mask)
                
        print(f"Best loss: {best_loss}")        
        return xnew.squeeze(0)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        self.X_obs, self.y_obs = totorch(X_obs, self.device), totorch(y_obs, self.device).reshape(-1)

        self.params['context_seq_len'] = len(y_obs)
        self.get_model()
        
        # Get sampled Triples dataset (x, I, C) 
        train_dataset = TriplesDataset(self.X_obs, self.y_obs, self.device, 'train')

        # Evaluate dataset
        eval_dataset = TriplesDataset(self.X_obs, self.y_obs, self.device, 'eval')
        
        # Train the model
        rootdir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(rootdir, "model_checkpoints/")
        os.makedirs(path, exist_ok=True)
        x_new = self.train_and_eval(train_dataset, eval_dataset, save_path = path)

        return x_new.cpu().numpy()







