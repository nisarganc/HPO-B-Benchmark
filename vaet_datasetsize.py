from math import comb

def calculate_context_combinations(N, first):
    if first:
        context_combinations = 2**N - 1
        dataset_triples = N*context_combinations
        return (context_combinations, dataset_triples) 
    else:
        context_combinations = int((2**N - 1)/2)+1
        dataset_triples = (N)*context_combinations+ 1 * (2**(N-1) - 1)
        return (context_combinations, dataset_triples)

first = True
context_list = []
dataset_list = []
initial_history = 5
trials = 50
print("History Length | Context combinations | Dataset triples")
for N in range(0, trials):
    context, triples = calculate_context_combinations(initial_history+N, first=first)
    first = False
    context_list.append(context)
    dataset_list.append(triples)
    print(f"{initial_history+N} | {context} | {triples}")

# plot context_list and dataset_list over N
import matplotlib.pyplot as plt
import numpy as np
N = np.arange(initial_history, initial_history+trials)
plt.plot(N, context_list, label="context combinations")
plt.plot(N, dataset_list, label="dataset triples")
plt.legend()
plt.xlabel("Number of history elements")
plt.ylabel("Size of dataset")
plt.savefig("Dataset_size.png")
plt.show()


