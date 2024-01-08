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
# ### SIR Model of infection spreading
# 
# [comment]:  S\rightarrowI\rightarrowR
# [comment]:  \overset{\alpha}{\underset{\beta}{\leftrightarrows}} 
# 
# From timestep $t$ to timestep $t+1$ each infected node infects its neighbors with probability $p_{ij}$*$\beta$. 
# Also, after infecting or not its neighbours, each node recovers with probability $\gamma$
# This means that each neighbor will transit from a $\textbf{Susceptible}$ state to an $\textbf{Infected}$ state.
# and that each initially infected neighbor transits from a $\textbf{Infected}$ state to an $\textbf{Recovered}$ state.
# 
#
# ## $$ S\overset{}{\underset{\beta \cdot p_{ij}}{\rightarrow}}I $$
# 
# 
# In this case, $\beta$ acts like a global rescaling time factor of the propagation.  
# By default we consider $\beta = 1$ and $\gamma = 0.1$ meaning that each node infects its neighbor with exactly the proability given by their proximity
# and that each infected node recovers after a given iteration with probability $\gamma$.

# %%
#Receives a NetworkX network and returns 
#a list with the percentage of nodes infected in each time step until the infection disappears (0%)
#beta is a global parameter of the transmission of infection from one node to another
#gamma is a global parameter that asserts the probability of any infected individual to become recovered
def sir_infection_spreading_alternate(net, seed_node, beta=1, gamma=0.1):
    type_of_net = str(type(net)).split('.')[-1][:-2]

    #Creating a copy of the network
    net_copy = eval('nx.'+type_of_net+'()')
    net_copy.add_edges_from(net.edges(data=True))

    #Total number of nodes
    N = len(net_copy.nodes)
    #assigning every node as susceptible
    nx.set_node_attributes(net, 's', name='state')
    #assigning the first initial infected node
    net.nodes[seed_node]['state']='i'
    #Initializing the Infected list
    infected, infected_percentages = [seed_node], [1*100/N] 
    #Initializing the Recovered list
    recovered, recovered_percentages = [], [0]
    #Initializing the time of infection for each node
    times_for_infection = dict([(node, None) if node != seed_node else (node, 0) for node in net_copy.nodes() ])
    
    t=0
    while len(infected)>0:
        t+=1
        new_infected, new_recovered = [], []
        #Here we (possibily) add elements to infected, so we iterate only in the initial elements of infected
        for ind in range(len(infected)):
            node = infected[ind]
            susceptible_neighbours = [n for n in list(net_copy.neighbors(node)) if net.nodes[n]['state'] == 's']
            for neighbour in susceptible_neighbours:
                rn = rd.uniform(0,1)
                p = net[node][neighbour]['proximity']
                #Check that that node hasn't been infected yet in this iteration
                if rn < beta*p and neighbour not in new_infected:
                    new_infected.append(neighbour)
                    net_copy.remove_edge(node, neighbour)
            rn = rd.uniform(0,1)
            if rn < gamma:
                new_recovered.append(node)
         
        for node in new_infected:
            times_for_infection.update({node: t})
            net.nodes[node]['state'] = 'i'
            infected.append(node)
        for node in new_recovered:
            #if node not in recovered:
            net.nodes[node]['state'] = 'r'
            infected.remove(node)
            recovered.append(node)
        infected_percentages.append(len(infected)*100/N)
        recovered_percentages.append(len(recovered)*100/N)

    return infected_percentages, recovered_percentages, times_for_infection

# %%
#Receives a NetworkX network and returns 
#a list with the percentage of nodes infected in each time step until the infection disappears (0%)
#beta is a global parameter of the transmission of infection from one node to another
#gamma is a global parameter that asserts the probability of any infected individual to become recovered
def sir_infection_spreading(net, seed_node, beta, gamma=1):
    type_of_net = str(type(net)).split('.')[-1][:-2]

    #Creating a copy of the network
    net_copy = eval('nx.'+type_of_net+'()')
    net_copy.add_edges_from(net.edges(data=True))

    #Total number of nodes
    N = len(net_copy.nodes)
    #assigning every node as susceptible
    nx.set_node_attributes(net, 's', name='state')
    #assigning the first initial infected node
    net.nodes[seed_node]['state']='i'
    #Initializing the Infected list
    infected, infected_percentages = [seed_node], [1*100/N] 
    #Initializing the Recovered list
    recovered, recovered_percentages = [], [0]
    #Initializing the time of infection for each node
    
    while len(infected)>0:
        new_infected, new_recovered = [], []
        #Here we (possibily) add elements to infected, so we iterate only in the initial elements of infected
        for ind in range(len(infected)):
            node = infected[ind]
            susceptible_neighbours = [n for n in list(net_copy.neighbors(node)) if net.nodes[n]['state'] == 's']
            for neighbour in susceptible_neighbours:
                rn = rd.uniform(0,1)
                p = net[node][neighbour]['proximity']
                #Check that that node hasn't been infected yet in this iteration
                if rn < beta*p and neighbour not in new_infected:
                    new_infected.append(neighbour)
                    net_copy.remove_edge(node, neighbour)
            rn = rd.uniform(0,1)
            if rn < gamma:
                new_recovered.append(node)
         
        for node in new_infected:
            net.nodes[node]['state'] = 'i'
            infected.append(node)
        for node in new_recovered:
            #if node not in recovered:
            net.nodes[node]['state'] = 'r'
            infected.remove(node)
            recovered.append(node)
        infected_percentages.append(len(infected)*100/N)
        recovered_percentages.append(len(recovered)*100/N)

    return infected_percentages, recovered_percentages, 

# %%
#From a list of percentages, we retrieve the time of the maximum percentage of infected nodes and the time of the end of the infection, which is the last index.
def get_r_inf(percentages):
    return percentages[-1]

# %%
def repeat_infection_spreading(net, seeds, n_runs, beta):
    #Repeats Infection Spreadings n_runs times for each seed in seeds
    #Returns a list for each metric; The elemnts of each metric list are lists with the metric values, 
    #each sublist corresponds to a seed, and each entry in the sublist corresponds a simulation metric
    r_infs = []
    for s in seeds:
        seed_r_infs = []
        
        for r in range(1, n_runs+1):
            inf_percentages, rec_percentages = sir_infection_spreading(net, s, beta)
            seed_r_inf = get_r_inf(rec_percentages)
            print(f'Seed:{s} | Repetition:{r} | r_inf:{seed_r_inf}')
            seed_r_infs.append(seed_r_inf)   

        r_infs.append(seed_r_infs)
    
    return r_infs

#%%
def get_data_from_subgraph(subgraph, seed_nodes, number_of_spreadings, beta, data_retrieved): 
    nr_seed_nodes = len(seed_nodes)
    if data_retrieved=='mean_std':
        if subgraph is None:
            nr_seed_nodes = len(seed_nodes)
            return [[np.nan for i in range(nr_seed_nodes)] for j in range(2)]
        else:
            r_infs = repeat_infection_spreading(subgraph, seed_nodes, number_of_spreadings, beta)

            r_infs_means_by_seed = np.mean(r_infs, axis=1).tolist()
            r_infs_stds_by_seed = np.std(r_infs, axis=1).tolist()
            
            data = [r_infs_means_by_seed, r_infs_stds_by_seed]
            return data
        
    elif data_retrieved=='r_inf_values':
        if subgraph is None:
            return [[np.nan for i in range(number_of_spreadings)] for j in range(nr_seed_nodes)] 
        else:
            r_infs = repeat_infection_spreading(subgraph, seed_nodes, number_of_spreadings, beta)
            return r_infs
    else:
        print('Invalid data_retrieved type. Please choose between "mean_std" and "t_values"')

#%%
def plot_sir_spreading(net, my_seed_node, beta, gamma=0.1):
    sir_ip, sir_rp = sir_infection_spreading(net, my_seed_node, beta, gamma)
    sir_timestamps = [n for n in range(len(sir_ip))]
    sir_sp = [ 100 - sir_ip_el - sir_rp_el for (sir_ip_el, sir_rp_el) in zip(sir_ip,sir_rp)]
    plt.scatter(sir_timestamps,sir_sp, color='b', marker='.', label='Susceptible')
    plt.scatter(sir_timestamps,sir_ip, color='r', marker='.', label='Infected')
    plt.scatter(sir_timestamps,sir_rp, color='g', marker='.', label='Recovered')
    plt.title('SIR Model')
    plt.xlabel('Discrete Time')
    plt.ylabel('Percentage of nodes (%)')
    plt.legend()
    