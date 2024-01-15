from genericpath import exists
import torch
from vaet_modules import VAET
import numpy as np
import os
import json
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    
    rootdir   = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(999)
    torch.manual_seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    path = os.path.join(rootdir,"../..", "hpob-data")

    with open(path+"/meta-validation-dataset.json", "r") as f:
        valid_data = json.load(f) 
        valid_data = valid_data[args.space]    

    with open(path+"/meta-train-dataset.json", "r") as f:
        train_data = json.load(f) 
        train_data = train_data[args.space]

    os.makedirs(os.path.join(rootdir,"model_checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(rootdir,"model_checkpoints", f"{args.space}")
    fsbo_model = VAET(train_data = train_data, valid_data = valid_data, checkpoint_path = checkpoint_path, args=args)
    fsbo_model.meta_train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # search space id
    parser.add_argument('--space', help='Search Space Id', type=str, default="5971")

    # Training params
    parser.add_argument('--lr', help='Learning Rate', type=float, default=0.0001)
    parser.add_argument('--max-epochs', help='Meta-Train max epochs', type=int, default=2000)
    # parser.add_argument('--batch-size', help='Meta-Train batch size', type=int, default=1)

    # Model Transformer params
    parser.add_argument('--transformer-model-dim', help='Model Transformer dimension', type=int, default=32)
    parser.add_argument('--transformer-layers', help='Model Transformer layers', type=int, default=4)
    parser.add_argument('--transformer-num-heads', help='Model Transformer heads', type=int, default=4)
    parser.add_argument('--transformer-dim-ffn', help='Model Transformer dim ffn', type=int, default=64)
    parser.add_argument('--transformer-pre-normalization', help='Model Transformer pre normalization', type=bool, default=False)

    # Model VAE params
    parser.add_argument('--enc-hidden-dim-vae', help='Model VAE encoder hidden dim', type=int, default=32)
    parser.add_argument('--dec-hidden-dim-vae', help='Model VAE decoder hidden dim', type=int, default=128)
    parser.add_argument('--latent-dim-vae', help='Model VAE latent dim', type=int, default=8)

    # Model configs
    parser.add_argument('--dropout', help='Model dropout', type=float, default=0.1)
    parser.add_argument('--init-fn', help='Model init fn', type=str, default='normal_init')

    args = parser.parse_args()

    main(args)