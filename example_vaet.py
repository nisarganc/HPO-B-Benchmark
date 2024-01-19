import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse

from hpob_handler import HPOBHandler
from methods.vaet.vaet_modules import generativeHPO


def main(args):

    hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/") 
    search_space_id = args.space                            # hpob_hdlr.get_search_spaces()
    dataset_ids =  hpob_hdlr.get_datasets(search_space_id)  # ['10093', '3954', '43', '34536', '9970', '6566']
    seeds = hpob_hdlr.get_seeds()
    dim = hpob_hdlr.get_search_space_dim(search_space_id)   # 16
    args.input_size = dim

    params = vars(args)

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
        torch_seed = 999

        for seed in seeds:
            #define the generative method as as HPO method
            method = generativeHPO(search_space_id, dataset_id, seed, params, torch_seed, verbose = False)
            #evaluate the HPO method
            acc = hpob_hdlr.evaluate_continuous(method, 
                                                search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                seed = seed,
                                                n_trials = params['trials'] )
            acc_per_method.append(acc)
            results[search_space_id][dataset_id][seed] = acc

            with open(results_file, 'w') as fp:
                json.dump(results, fp)

        plt.plot(np.array(acc_per_method).mean(axis=0))

    plt.legend(dataset_ids)
    plt.savefig("vaet_accresults.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # search space id
    parser.add_argument('--space', help='Search Space Id', type=str, default="5971")
    parser.add_argument('--trials', help='Number of trials', type=str, default=50)

    # Training params
    parser.add_argument('--lr', help='Learning Rate', type=float, default=0.001)
    parser.add_argument('--max_epochs', help='Meta-Train max epochs', type=int, default=150)
    parser.add_argument('--batch_size', help='Meta-Train batch size', type=int, default=75)
    parser.add_argument('--max_hlength', help='Meta-Train dataset size', type=int, default=10)

    # Model Transformer params
    parser.add_argument('--transformer-model-dim', help='Model Transformer dimension', type=int, default=32)
    parser.add_argument('--transformer-layers', help='Model Transformer layers', type=int, default=4)
    parser.add_argument('--transformer-num-heads', help='Model Transformer heads', type=int, default=4)
    parser.add_argument('--transformer-dim-ffn', help='Model Transformer dim ffn', type=int, default=64)
    parser.add_argument('--transformer-pre-normalization', help='Model Transformer pre normalization', type=bool, default=True)

    # Model VAE params
    parser.add_argument('--enc-hidden-dim-vae', help='Model VAE encoder hidden dim', type=int, default=256)
    parser.add_argument('--dec-hidden-dim-vae', help='Model VAE decoder hidden dim', type=int, default=256)
    parser.add_argument('--latent-dim-vae', help='Model VAE latent dim', type=int, default=8)

    # Model configs
    parser.add_argument('--dropout', help='Model dropout', type=float, default=0.1)
    parser.add_argument('--init-fn', help='Model init fn', type=str, default='xavier_init')

    args = parser.parse_args()

    main(args)    