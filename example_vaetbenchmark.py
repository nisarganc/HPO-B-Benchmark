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
    n_trials = 10
    torch_seed = 999

    benchmark_plotter  = BenchmarkPlotter(experiments = experiments, 
                                          seeds=["test1", "test2", "test3", "test4"],
                                          name = name,
                                          n_trials = n_trials,
                                            results_path = results_path, 
                                            output_path = output_path, 
                                            data_path = data_path, 
                                            search_spaces=['5971'])

    benchmark_plotter.generate_rank_and_regret()
    benchmark_plotter.generate_plots_per_search_space()



