'''
util functions required for the model and training of the model.
Samples all possible Triples from history H and create dataset class and collate function for the dataloader.
'''
import torch
import numpy as np
from itertools import combinations
import torch.nn as nn

import random
from math import comb as comb

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def totorch(x, device):
    return torch.Tensor(x).to(device)  

def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
  """Intialization of layers with normal distribution with mean and bias"""
  classname = layer.__class__.__name__
  if classname.find('Linear') != -1:
    layer.weight.data.normal_(mean_, sd_)
    if norm_bias:
      layer.bias.data.normal_(bias, 0.05)
    else:
      layer.bias.data.fill_(bias)

def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
  classname = layer.__class__.__name__
  if classname.find('Linear')!=-1:
    nn.init.xavier_uniform_(layer.weight.data)
    if norm_bias:
      layer.bias.data.normal_(0, 0.05)
    else:
      layer.bias.data.zero_()

def weight_init(
    module, 
    mean_= 0, 
    sd_= 0.004, 
    bias = 0.0, 
    norm_bias = False, 
    init_fn_= normal_init_):
    moduleclass = module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    init_fn_(l, mean_, sd_, bias, norm_bias)
            else:
                init_fn_(layer, mean_, sd_, bias, norm_bias)
    except TypeError:
        init_fn_(module, mean_, sd_, bias, norm_bias)

class TriplesDataset(torch.utils.data.Dataset):
    def __init__(self, params, mode='train'):
        self.mode = mode
        self.dimension = params['input_size']
        self.max_hlength = params['max_hlength']
        self.device = device()
        self.all_combinations = []
        self.all_masks = []
        self.triples = []

    def load_history(self, x_obs, y_obs):
        self.x_obs = x_obs
        self.y_obs = y_obs 
        self.hlength = len(self.x_obs) 

        self.all_combinations = []
        self.all_masks = []
        self.triples = []

        print(f"Inital observation History: {self.hlength}")
        for subset_size in range(1, self.hlength + 1):
            combos = combinations(zip(self.x_obs, self.y_obs), subset_size)
            combos = [list(combination) for combination in combos]
            
            # pad with zeros to keep a fixed size 
            combos = [combination + 
                      [(torch.zeros(self.dimension).to(self.device), torch.tensor(0.0, device=self.device))] * (self.max_hlength - subset_size) for combination in combos]

            # create a mask to distinguish between observed and padded elements in C
            mask = [[0.0] * subset_size + [1.0] * (self.max_hlength - subset_size)] * len(combos)
            mask = torch.tensor(mask, device=self.device)

            # convert tuples and mask to tensors by concatenating x and y
            combos = [torch.stack([torch.cat([x_i.clone().detach().to(self.device), y_i.clone().detach().unsqueeze(0).to(self.device)]) for x_i, y_i in combination], dim=0) for combination in combos]

            self.all_combinations.extend(combos) 
            self.all_masks.extend(mask)
        print(f"Initial context combinations: {len(self.all_combinations)}") 

        for i in range(self.hlength):
            x = self.x_obs[i]
            y = self.y_obs[i]
        
            # determine I for each combination C
            for C, mask in zip(self.all_combinations, self.all_masks):
                I = torch.tensor(int(all(y > xy_i[-1] for xy_i in C)), dtype=torch.int16, device=self.device)
                self.triples.append((x, I, C, mask)) 
                
        print(f"Dataset Triples: {len(self.triples)}")

    def update_history(self, new_x, new_y):
        """ Generate new triples with the new x and y observations """ 
        self.new_combinations = []
        self.new_masks = []
        
        if self.hlength < self.max_hlength:
            self.x_obs = torch.cat((self.x_obs, new_x.unsqueeze(0)), dim=0)
            self.y_obs = torch.cat((self.y_obs, new_y), dim=0)
            self.hlength += 1   # 15
            self.triples = []
            self.update_all(new_x, new_y)
        else:
            # remove one x and y from the history whose y is the smallest
            min_y = torch.min(self.y_obs)
            min_y_idx = torch.argmin(self.y_obs)
            self.x_obs = torch.cat((self.x_obs[:min_y_idx], self.x_obs[min_y_idx+1:]), dim=0)
            self.y_obs = torch.cat((self.y_obs[:min_y_idx], self.y_obs[min_y_idx+1:]), dim=0)

            # add the new x and y to the history
            self.x_obs = torch.cat((self.x_obs, new_x.unsqueeze(0)), dim=0)
            self.y_obs = torch.cat((self.y_obs, new_y), dim=0)
            self.triples = [] 
            self.update_all(new_x, new_y)

    def update_all(self, new_x, new_y):
        """ Generate new combinations that include the new element """
        print(f"Updated observation History: {self.hlength}")

        new_combos = []

        for subset_size in range(1, self.hlength + 1):
            combos = combinations(zip(self.x_obs, self.y_obs), subset_size)
            combos = [ list(combination) for combination in combos ]

            new_combos = []
            for context in combos:
                for x_i, y_i in context:
                    if torch.all(torch.eq(new_x, x_i)) and torch.eq(new_y, y_i):
                        new_combos.append(context) 
                        break
                          
            new_combos = [combination + 
                          [(torch.zeros(self.dimension).to(self.device), torch.tensor(0.0, device=self.device))] * (self.max_hlength - subset_size) for combination in new_combos]

            mask = [[0.0] * subset_size + [1.0] * (self.max_hlength - subset_size)] * len(new_combos)
            mask = torch.tensor(mask, device=self.device)

            new_combos = [torch.stack([torch.cat([x_i.clone().detach().to(self.device), y_i.clone().detach().unsqueeze(0).to(self.device)]) for x_i, y_i in combination], dim=0) for combination in new_combos]

            self.new_combinations.extend(new_combos)
            self.new_masks.extend(mask)
        print(f"New context combinations: {len(self.new_combinations)}") 

        for i in range(self.hlength):
            x = self.x_obs[i]
            y = self.y_obs[i]
            for C, mask in zip(self.new_combinations, self.new_masks):
                I = torch.tensor(int(all(y > xy_i[-1] for xy_i in C)), dtype=torch.int16, device=self.device)
                self.triples.append((x, I, C, mask))  

        x = new_x
        y = new_y
        for C, mask in zip(self.all_combinations, self.all_masks):
            I = torch.tensor(int(all(y > xy_i[-1] for xy_i in C)), dtype=torch.int16, device=self.device)
            self.triples.append((x, I, C, mask))

        self.all_combinations.extend(self.new_combinations)
        self.all_masks.extend(self.new_masks)
        print(f"Total context combinations: {len(self.all_combinations)}")
        print(f"Dataset Triples: {len(self.triples)}") 


    def load_eval_data(self):
        self.triples = []
        H = torch.stack([torch.cat([x_i.clone().detach().to(self.device), y_i.clone().detach().unsqueeze(0).to(self.device)]) for x_i, y_i in zip(self.x_obs, self.y_obs)], dim=0)
        H = torch.cat([H, torch.zeros(self.max_hlength - self.hlength, self.dimension + 1).to(self.device)], dim=0)
        mask = torch.tensor([0.0] * self.hlength, device=self.device)
        mask = torch.cat([mask, torch.ones(self.max_hlength - self.hlength, device=self.device)], dim=0)
        self.triples.append((
            torch.zeros(self.dimension).to(self.device), # dummy x
            torch.tensor(1, dtype=torch.int16, device=self.device),  # I=1
            H, # C=H 
            mask # mask    
        ))
        
    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return (
            self.triples[idx][0],
            self.triples[idx][1],
            self.triples[idx][2],
            self.triples[idx][3]
            )

# collate function for the dataloader
def collate_fn(batch):
    x, I, C, mask = zip(*batch)
    x = torch.stack(x, dim=0)
    I = torch.stack(I, dim=0)
    C = torch.stack(C, dim=0)

    mask = torch.stack(mask, dim=0)
    batch_ = {'x': x, 'I': I, 'C': C, 'mask': mask}
    return batch_



    # def update_samples(self, new_x, new_y):
    #     # Generate new combinations that include the new element
    #     print(f"Updated observation History: {self.hlength}")        
    #     max_combinations = 16384
    #     for subset_size in range(1, self.hlength + 1):
            
    #         indices = range(len(self.hlength))
           
    #         if len(indices) >= subset_size:
    #             sampled_indices = random.sample(list(combinations(indices, subset_size)), min(comb(len(indices), subset_size), max_combinations))

    #             # Create combinations using sampled indices
    #             for combo_indices in sampled_indices:
    #                 combo = [(self.x_obs[i], self.y_obs[i]) for i in combo_indices]
    #                 combo = combo + [(torch.zeros(self.dimension).to(self.device), torch.tensor(0.0, device=self.device))] * (self.max_contexts - subset_size)
    #                 mask = [0.0] * subset_size + [1.0] * (self.max_contexts - subset_size)
    #                 mask = torch.tensor(mask, device=self.device)
    #                 combo = torch.stack([torch.cat([x_i.clone().detach().to(self.device), y_i.clone().detach().unsqueeze(0).to(self.device)]) for x_i, y_i in combo], dim=0)

    #                 self.new_combinations.append(combo)
    #                 self.new_masks.append(mask)
    #     print(f"New context combinations: {len(self.new_combinations)}")


# if min_y == new_y and torch.all(torch.eq(self.x_obs[min_y_idx], new_x)):