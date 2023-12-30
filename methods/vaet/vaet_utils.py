'''
util functions required for the model and training of the model.
Samples all possible Triples from history H and create dataset class and collate function for the dataloader.
'''
import torch
import numpy as np
from itertools import combinations
import torch.nn as nn

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
    def __init__(self, x_obs, y_obs, device, mode='train'):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.mode = mode
        self.dimension = x_obs.shape[1]
        self.device = device
        self.load_data()

    def load_data(self):
        self.triples = []
        context_length = len(self.x_obs)

        if self.mode == 'train':
            for i in range(len(self.x_obs)):
                x = self.x_obs[i]
                y = self.y_obs[i]
                
                # All combinations C from one element to all elements in H (list of tuples)
                all_combinations = []
                all_masks = []

                for subset_size in range(1, len(self.x_obs) + 1):
                    combos = combinations(zip(self.x_obs, self.y_obs), subset_size)
                    combos = [list(combination) for combination in combos]

                    # pad with zeros to keep a fixed size of context_length
                    combos = [combination + [(torch.zeros(self.dimension).to(self.device), torch.tensor(0.0, device=self.device))] * (context_length - subset_size) for combination in combos]

                    # create a mask to distinguish between observed and padded elements in C
                    mask = [[0.0] * subset_size + [1.0] * (context_length - subset_size)] * len(combos)
                    mask = torch.tensor(mask, device=self.device)

                    # convert tuples and mask to tensors by concatenating x and y
                    combos = [torch.stack([torch.cat([torch.tensor(x_i, device=self.device), torch.tensor(y_i, device=self.device).unsqueeze(0)]) for x_i, y_i in combination], dim=0) for combination in combos]

                    all_combinations.extend(combos) 
                    all_masks.extend(mask)

            # determine I for each combination C
            for C, mask in zip(all_combinations, all_masks):
                I = torch.tensor(int(all(y > xy_i[-1] for xy_i in C)), dtype=torch.int16, device=self.device)
                self.triples.append((x, I, C, mask))  
        else:
            # For 'eval' mode
            self.triples.append((
                self.x_obs[0], # dummy x
                torch.tensor(1, dtype=torch.int16, device=self.device),  # I=1
                torch.stack([torch.cat([torch.tensor(x_i, device=self.device), torch.tensor(y_i, device=self.device).unsqueeze(0)]) for x_i, y_i in zip(self.x_obs, self.y_obs)], dim=0),
                torch.tensor([0.0] * context_length, device=self.device)
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
