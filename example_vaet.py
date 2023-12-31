import matplotlib.pyplot as plt
import numpy as np
import os
import json

from hpob_handler import HPOBHandler
from methods.vaet.vaet_modules import generativeHPO

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/") 
search_space_id = "5971" # hpob_hdlr.get_search_spaces()
dataset_ids =  hpob_hdlr.get_datasets(search_space_id)  # ['10093', '3954', '43', '34536', '9970', '6566']
seeds = ["test0", "test1", "test2", "test3", "test4"]

dim = hpob_hdlr.get_search_space_dim(search_space_id)

# create a json file within results folder to store the results
rootdir = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(rootdir, "results")
os.makedirs(results_dir,exist_ok=True)
results_file = os.path.join(results_dir, "VAET.json")
open(results_file, 'w').close()

results = {search_space_id: {}}

for dataset_id in dataset_ids:
    acc_per_method = []
    results[search_space_id][dataset_id] = {}

    for seed in seeds:
        torch_seed = 999

        params = {
            # Input data params
            'hyperparam_dim': dim, 

            # Training params
            'lr': 0.001,
            'epochs': 1000,
            'batch_size': 5,

            # Model Transformer params
            'transformer_model_dim': 128, 
            'transformer_layers': 4,
            'transformer_num_heads': 4,
            'transformer_dim_ffn': 256,
            'transformer_pre_normalization': False,

            # Model VAE params
            'hidden_dim_vae': 256,
            'latent_dim_vae': 512,
            
            # Model configs 
            'dropout': 0.1,
            'init_fn': 'normal_init',   
        }

        #define the generative method as as HPO method
        method = generativeHPO(params, torch_seed, verbose = True)

        #evaluate the HPO method
        acc = hpob_hdlr.evaluate_continuous(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                seed = seed,
                                                n_trials = 50 )
        acc_per_method.append(acc)
        results[search_space_id][dataset_id][seed] = acc
    plt.plot(np.array(acc_per_method).mean(axis=0))

# save the results and write them to the json file
with open(results_file, 'w') as fp:
    json.dump(results, fp)  
plt.legend(dataset_ids)
plt.show()
