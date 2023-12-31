import numpy as np
import json
from benchmark_plot import BenchmarkPlotter
from methods.vaet.vaet_modules import generativeHPO

if __name__ == "__main__":

    data_path = "hpob-data/"
    results_path = "results/"
    output_path = "plots/"
    name = "vaet_benchmark"
    new_method_name = "VAET.json"
    experiments = ["Random", "DGP", "GP", "VAET"]
    n_trials = 50
    torch_seed = 999

    benchmark_plotter  = BenchmarkPlotter(experiments = experiments, 
                                          name = name,
                                          n_trials = n_trials,
                                          results_path = results_path, 
                                          output_path = output_path, 
                                          data_path = data_path, 
                                          search_spaces=['5971'])

    benchmark_plotter.generate_rank_and_regret()
    benchmark_plotter.generate_aggregated_plots()



