# %% [markdown]
# # Inspecting metric backbone properties in Directed Networks  
#   
# ##### Author: M. Bernardo G. Pereira  
#   
# This work is inspired by the article:  
#   
# *Brattig Correia R, Barrat A, Rocha LM (2023) Contact networks have small metric backbones that maintain community structure and are primary transmission subgraphs. PLoS Comput Biol 19(2): e1010854. https://doi.org/10.1371/journal.pcbi.1010854*  
#   
# However, in this case, the networks being analysed have directed edges, as in the article:
# 
# *Costa, F.X., Correia, R.B., Rocha, L.M. (2023). The Distance Backbone of Directed Networks. In: Cherifi, H., Mantegna, R.N., Rocha, L.M., Cherifi, C., Micciche, S. (eds) Complex Networks and Their Applications XI. COMPLEX NETWORKS 2016 2022. Studies in Computational Intelligence, vol 1078. Springer, Cham. https://doi.org/10.1007/978-3-031-21131-7_11*

# %%
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import distanceclosure as dc
import matplotlib.pyplot as plt

# %% [markdown]
# # Part 3 - Modelling

# %% [markdown]
# ### SI Model of infection spreading
# 
# [comment]:  S\rightarrowI
# [comment]:  \overset{\alpha}{\underset{\beta}{\leftrightarrows}} 
# 
# From timestep $t$ to timestep $t+1$ each infected node infects its neighbors with probability $p_{ij}$.  
# This means that each neighbor will transit from a $\textbf{Susceptible}$ state to an $\textbf{Infected}$ state.
# 
# 
# ## $$ S\overset{}{\underset{\beta \cdot p_{ij}}{\rightarrow}}I $$
# 
# 
# In this case, $\beta$ acts like a global rescaling factor of the propagation.  
# By default we consider $\beta = 0.9 \cdot p_{max}$ where $p_{max}$ is the largest proximity weight of the original network.

# %%
#Receives a NetworkX network and returns 
#a list with the percentage of nodes infected in each time step until 100%
#beta is a global parameter of the transmission of infection from one node to another [0,1]

#In the article, beta is 0.9*mp where mp is the maximum value of p_{ij}
def si_infection_spreading(net, seed_node, beta=3):
    type_of_net = str(type(net)).split('.')[-1][:-2]

    #Creating a copy of the network
    net_copy = eval('nx.'+type_of_net+'()')
    net_copy.add_edges_from(net.edges(data=True))

    #assigning every node as susceptible
    nx.set_node_attributes(net, 's', name='state')
    #assigning the input seed_node as infected
    net.nodes[seed_node]['state'] = 'i'
    infected = [seed_node]
    #Total number of nodes
    N = len(net.nodes) 
    #Initially the percentage of nodes infected is 1/(number of nodes)
    percentages = [1*100/N]
    while len(infected) < N :        
        for ind in range(len(infected)):
            node = infected[ind]
            susceptible_neighbors = [n for n in list(net_copy.neighbors(node)) if net.nodes[n]['state'] == 's']      
            for neighbor in susceptible_neighbors:
                rn = rd.uniform(0, 1)
                p = net[node][neighbor]['proximity']
                if rn < beta*p:                  
                    infected.append(neighbor)
                    net_copy.remove_edge(node, neighbor)
                    net.nodes[neighbor]['state'] = 'i'
        perc = len(infected)/N
        percentages.append(perc*100)
    return percentages

# %%
#From a list of percentages, we retrieve the time of the maximum percentage of infected nodes and the time of the end of the infection, which is the last index.
def get_t_half_t_all(percentages):
    t_all = len(percentages)-1
    switch = True
    i=0
    while switch:
        if percentages[i]>=50:
            t_half = i
            switch = False
        i+=1
    return t_half, t_all

# %%
def repeat_infection_spreading(net, seeds, n_runs):
    t_halfs, t_alls = [], []
    for s in seeds:
        seed_t_halfs, seed_t_alls = [], []
        for r in range(1, n_runs+1):
            percentages = si_infection_spreading(net, seed_node=s)
            seed_t_half, seed_t_all = get_t_half_t_all(percentages)
            print(f'Seed:{s} | Repetition:{r} | t_half:{seed_t_half} | t_all:{seed_t_all}')
            seed_t_halfs.append(seed_t_half)
            seed_t_alls.append(seed_t_all)                
        t_halfs.append(seed_t_halfs)
        t_alls.append(seed_t_alls)
    return t_halfs, t_alls

#%%
def get_data_from_subgraph(subgraph, seed_nodes, number_of_spreadings, data_retrieved):
    nr_seed_nodes = len(seed_nodes)
    if data_retrieved == 'mean_std':
        if subgraph is None:
            nr_seed_nodes = len(seed_nodes)
            return [[np.nan for i in range(nr_seed_nodes)] for j in range(4)]
        else:
            t_halfs, t_alls = repeat_infection_spreading(subgraph, seeds=seed_nodes, n_runs=number_of_spreadings)

            t_halfs_means_by_seed = np.mean(t_halfs, axis=1).tolist()
            t_halfs_stds_by_seed = np.std(t_halfs, axis=1).tolist()
            t_alls_means_by_seed = np.mean(t_alls, axis=1).tolist()
            t_alls_stds_by_seed = np.std(t_alls, axis=1).tolist()

            data = [t_halfs_means_by_seed, t_halfs_stds_by_seed, t_alls_means_by_seed, t_alls_stds_by_seed]
            return data

    elif data_retrieved == 't_values':
        if subgraph is None:
            return [[[np.nan for i in range(number_of_spreadings)] for j in range(nr_seed_nodes)] for k in range(2)]
        else:
            t_halfs, t_alls = repeat_infection_spreading(subgraph, seeds=seed_nodes, n_runs=number_of_spreadings)
            return [t_halfs, t_alls]
    else:
        print('Error: data_retrieved type must be either mean_std or t_values')
        
#%%
def plot_si_spreading(net, my_seed_node, beta):
    si_ip = si_infection_spreading(net, my_seed_node, beta)
    si_timestamps = [n for n in range(len(si_ip))]
    si_sp = [ 100 - si_ip_el for si_ip_el in si_ip]
    plt.scatter(si_timestamps, si_ip, color='r', marker='.', label='Infected')
    plt.scatter(si_timestamps, si_sp, color='b', marker='.', label='Susceptible')
    plt.title('SI Model')
    plt.xlabel('Discrete Time')
    plt.ylabel('Percentage of nodes (%)')
    plt.legend()
    