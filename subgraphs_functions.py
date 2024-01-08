# %% [markdown]
# # Inspecting distance backbone properties in Directed Networks  
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
import pandas as pd
import random as rd
import numpy as np
import networkx as nx
import copy
import pickle

# %%
def number_appropriate_connected_components(type_of_net, net):
    if type_of_net == 'DiGraph':
         res = nx.number_strongly_connected_components(net)
    else:
        res = nx.number_connected_components(net)
    return res
# %% [markdown]
# ## Create Intermediate Graphs: from Original Graph size to Backbone size

# %% [markdown]
# ### Create backbone random subgraphs
#
# To generate a backbone random subgraphs from any network we studied here, we fix the backbone edges, then we add x % of edges 
# at random from the remaining edges.
def backbone_random_subgraphs(dist_net, list_nr_edges, n):
    #Receives:
    # - dist_net : a network with distances and semi-triangular distortion (i.e. distances computed with a certain phi)
    # - list_percentages : a list with values of percentages of edges to add to the initial metric backbone subgraph
    #   typically in the form of [0,...,100]
    # - n : number of different random realizations of networks to create for each percentage value
    print('Producing Backbone Random Subgraphs')
    type_of_net = str(type(dist_net)).split('.')[-1][:-2]
    all_edges = list(dist_net.edges(data=True))
    backbone_edges = [edge for edge in all_edges if edge[-1]['distortion']==1]
    non_backbone_edges = [edge for edge in all_edges if edge[-1]['distortion']>1]
    
    nr_backbone_edges = len(backbone_edges)
    #nr_non_backbone_edges = len(non_backbone_edges)

    nrs_of_non_backbone_edges = [n-nr_backbone_edges for n in list_nr_edges]
    
    subgraphs = []

    for k in range(n):
        new_iter = []
        for i in nrs_of_non_backbone_edges:
            added_edges = rd.sample(non_backbone_edges, i)
            graph = eval('nx.'+type_of_net)(backbone_edges+added_edges)
            new_iter.append(graph)
        subgraphs.append(new_iter)
            
    return subgraphs

# %% [markdown]
# ## Create threshold random subgraphs
# 
# To generate a thresholded random subgraph from any network we studied here, we rank and remove the weakest edges 
# (i.e., edges with low proximity or large distance) of the original graph until we reach the same number 
# of edges as in the metric backbone, then we add x % of edges at random from the remaining edges.

def threshold_random_subgraphs(dist_net, list_nr_edges, n): 
    #Receives:
    # - dist_net : a network with distances and semi-triangular distortion (i.e. distances computed with a certain phi)
    # - list_percentages : a list with values of percentages of edges to add to the initial metric backbone subgraph
    #   typically in the form of [0,...,100]
    # - n : number of different random realizations of networks to create for each percentage value
    print('Producing Threshold Random Subgraphs')
    type_of_net = str(type(dist_net)).split('.')[-1][:-2]
    all_edges = list(dist_net.edges(data=True))
    all_edges.sort(key = lambda edge: edge[-1]['proximity'], reverse=True)

    nr_backbone_edges = len([edge for edge in all_edges if edge[-1]['distortion']==1])
    threshold_edges = all_edges[:nr_backbone_edges]
    non_threshold_edges = all_edges[nr_backbone_edges:]
    nr_non_threshold_edges = len(non_threshold_edges)

    nrs_of_added_edges = [n-nr_backbone_edges for n in list_nr_edges]
   
    subgraphs = []

    for k in range(n):
        new_iter = []
        rd.shuffle(non_threshold_edges)
        for i in nrs_of_added_edges:
            graph = eval('nx.'+type_of_net+'()')
            graph.add_nodes_from(dist_net.nodes)
            added_edges = non_threshold_edges[:i]
            graph.add_edges_from(threshold_edges+added_edges)
            if number_appropriate_connected_components(type_of_net, graph) == 1:
                new_iter.append(graph)
            else:
                new_iter.append(None)
        subgraphs.append(new_iter)
    
    return subgraphs
# %% [markdown]
# ## Create random subgraphs
# 
# To generate a random subgraph, we simply remove edges at random from the original graph until we reach the same number of edges as in the backbone.
def random_random_subgraphs(dist_net, list_nr_edges, n): 
    #Receives:
    # - dist_net : a network with distances and semi-triangular distortion (i.e. distances computed with a certain phi)
    # - list_percentages : a list with values of percentages of edges to add to the initial metric backbone subgraph
    #   typically in the form of [0,...,100]
    # - n : number of different random realizations of networks to create for each percentage value
    print('Producing Random Random Subgraphs')
    type_of_net = str(type(dist_net)).split('.')[-1][:-2]
    all_edges = list(dist_net.edges(data=True))

    nr_backbone_edges = len([edge for edge in all_edges if edge[-1]['distortion']==1])
    nr_non_backbone_edges = len(all_edges) - nr_backbone_edges
    nrs_of_non_baseline_edges = [n-nr_backbone_edges for n in list_nr_edges]
    
    subgraphs = []
    for k in range(n):
        new_iter = []
        rd.shuffle(all_edges)
        for i in nrs_of_non_baseline_edges:
            graph = eval('nx.'+type_of_net)()
            graph.add_nodes_from(dist_net.nodes)
            nr_subgraph_edges = nr_backbone_edges + i
            graph.add_edges_from(all_edges[:nr_subgraph_edges])
            if number_appropriate_connected_components(type_of_net, graph) == 1:
                new_iter.append(graph)
            else:
                new_iter.append(None)
        subgraphs.append(new_iter)
    return subgraphs

#Note for posteriority:
#We need to add the nodes when constructiong subgraphs because if we just add the edges, the graph might only have just one cc and not consider some nodes 
#that are left out because they dont have sufficiently strong connections to be in the threshold+added edges graph
#This took me too long to realize and was why I had to make that elaborate implementation that was based on removing edges instead of adding them
#However, this is much more efficient

# %% [markdown]
# Create Distortion Threshold subgraphs
# 
# To generate these subgraphs we sort the edges by distortion and remove the edges with larger distortion of the original graph until 
# we reach the same number of edges as in the backbone.
def threshold_proximity_subgraphs(dist_net, list_nr_edges):
    #Receives:
    # - net : a network with proximities and semi-triangular distortion (i.e. distances computed with a certain phi)
    # - list_percentages : a list with values of percentages of edges to add to the initial random subgraph of the size of the backbone
    #   typically in the form of [0,...,100]
    #Returns:
    # - subgraphs : list of subgraphs of the original network, created by thresholding the proximity of the edges and where the 0% subgraph is the 
    # proximity threshold subgraph of the same size as the backbone inputed
    print('Producing Threshold Proximity Subgraphs')
    type_of_net = str(type(dist_net)).split('.')[-1][:-2]
    all_edges = list(dist_net.edges(data=True))
    all_edges.sort(key = lambda edge: edge[-1]['proximity'], reverse=True)
    nr_backbone_edges = len([edge for edge in all_edges if edge[-1]['distortion']==1])
    nr_total_edges = len(all_edges)
    nr_non_backbone_edges = nr_total_edges - nr_backbone_edges

    nrs_of_edges = list_nr_edges

    subgraphs = []    
    for i in nrs_of_edges:
        graph = eval('nx.'+type_of_net+'()')
        graph.add_nodes_from(dist_net.nodes)
        graph.add_edges_from(all_edges[:i])
        if number_appropriate_connected_components(type_of_net, graph) == 1:
            subgraphs.append(graph)
        else: 
            subgraphs.append(None)
    return [subgraphs]

# %% [markdown]
# Create Distortion Threshold subgraphs
# 
# To generate these subgraphs we sort the edges by distortion and remove the edges with larger distortion of the original graph until we reach the same number of edges as in the backbone.
def threshold_distortion_subgraphs(dist_net, list_nr_edges):
    #Receives:
    # - net : a network with distances and semi-triangular distortion (i.e. distances computed with a certain phi)
    # - list_percentages : a list with values of percentages of edges to add to the initial random subgraph of the size of the backbone
    #   typically in the form of [0,...,100]
    #Returns:
    # - subgraphs : list of subgraphs of the original network, created by thresholding the distortion of the edges in the values of the list percentages
    print('Producing Threshold Distortion Subgraphs')
    type_of_net = str(type(dist_net)).split('.')[-1][:-2]
    all_edges = list(dist_net.edges(data=True))
    all_edges.sort(key = lambda edge: edge[-1]['distortion'])
    nr_backbone_edges = len([edge for edge in all_edges if edge[-1]['distortion']==1])
    nr_total_edges = len(all_edges)
    nr_non_backbone_edges = nr_total_edges - nr_backbone_edges

    nrs_of_edges = list_nr_edges
    subgraphs = []
    for i in nrs_of_edges:
        graph = eval('nx.'+type_of_net)(all_edges[:i])
        subgraphs.append(graph)

    return [subgraphs]

# %%
# #### Outputting a dataframe with the number of edges in each different subgraph
def data_nr_edges_of_subgraphs(net, list_percentages):
    #Returns a dataframe with 3 lines ('nr_edges_ms', 'nr_edges_ts', 'nr_edges_rs') and with columns for each percentage value entry has the number of edge
    
    ms_net = backbone_random_subgraphs(net, list_percentages, n=1)
    ts_net = threshold_random_subgraphs(net, list_percentages, n=1)
    rs_net = random_random_subgraphs(net, list_percentages, n=1)

    len_edges_ms = [len(n.edges) for n in ms_net]
    len_edges_ts = [len(n.edges) for n in ts_net]
    len_edges_rs = [len(n.edges) for n in rs_net]

    lst = [reversed(len_edges_ms), reversed(len_edges_ts), reversed(len_edges_rs)]
        
    df = pd.DataFrame(lst, index =['nr_edges_brs', 'nr_edges_trs', 'nr_edges_rrs'], columns = reversed(list_percentages))
    df = df[sorted(df)]
    return df

# %%
def get_fraction_connected_subgraphs(subgraphs):
    nr_realizations = len(subgraphs)
    nr_net_sizes = len(subgraphs[0])
    fractions = [ 1 - (([subgraphs[j][i] for j in range(nr_realizations)]).count(None)/nr_realizations) for i in range(nr_net_sizes)]
    return fractions

def get_fraction_connected_subgraphs_old(subgraphs):
    nr_net_sizes = len(subgraphs[0])
    nr_realizations = len(subgraphs)
    fractions = []
    for i in range(nr_net_sizes): #Network Size
        s = 0
        for j in range(nr_realizations): #Network Realization
            if subgraphs[j][i] == None:
                s+=1
        fractions.append(1 - s/nr_realizations)
    return fractions

# %%
#When making plots that include several backbones that are included in eachother (like UM c M c P C G) we need to create a list of percentage for every backbone
# so that in the end the list of net_sizes are included in eachother in the same way as the backbones.
# All this is needed because the subgraphs functions all take as inputs a list of percentages for the networks sizes.
def get_best_subgraphs_sizes(dist_nets_names, dist_nets_list):
    #Receives:
    # - dist_nets_list : a list of networks with distances and semi-triangular distortion (i.e. distances computed with a certain phi)
    #Returns:
    # -best_net_sizes : a list of network sizes that are the same for all the networks in dist_nets_list
    # -dict_backbones_percentages : a dictionary with the percentages of edges that are in the subgraphs for each backbone subgraphs

    nr_backbones = len(dist_nets_list)
    dict_backbones_sizes = dict([(dist_nets_names[i], len([edge for edge in list(dist_nets_list[i].edges(data=True)) if edge[-1]['distortion']==1])) for i in range(nr_backbones)])
    nr_backbones = len(dict_backbones_sizes)
    min_subgraph_size = min(dict_backbones_sizes.values())
    max_subgraph_size = len(dist_nets_list[0].edges)
    range_subgraph_sizes = max_subgraph_size - min_subgraph_size
    optimal_interval_size = range_subgraph_sizes//(11-nr_backbones)
    best_net_sizes = [n for n in range(min_subgraph_size, max_subgraph_size+1) if (n%optimal_interval_size==0) or (n in dict_backbones_sizes.values()) or (n==max_subgraph_size)]

    dict_backbones_sizes_index = dict([(dist_nets_names[i], best_net_sizes.index(dict_backbones_sizes[dist_nets_names[i]])) for i in range(nr_backbones)])
    dict_backbones_sublist = dict([(dist_nets_names[i], best_net_sizes[(dict_backbones_sizes_index[dist_nets_names[i]]):] ) for i in range(nr_backbones)])

    dict_backbones_percentages = dict([(dist_nets_names[i], [(el - dict_backbones_sizes[dist_nets_names[i]])*100/(max_subgraph_size-dict_backbones_sizes[dist_nets_names[i]]) for el in dict_backbones_sublist[dist_nets_names[i]]]) for i in range(nr_backbones)])

    return best_net_sizes, dict_backbones_percentages

def get_best_net_sizes_for_k_backbones_family(net_name, net_sizes, phi):
    df = pd.read_html(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/{net_name}_{phi}_powers_nr_edges.html')[0]
    sizes=[]
    rel_powers=[]
    for power in df.columns[1:]:
        cur_size = int(df.loc[:,power])
        if cur_size not in sizes:
            sizes.append(cur_size)
            rel_powers.append(power)
    net_sizes=[326, 426, 603, 623, 915, 1830, 2745, 3660, 4369, 4575, 5490, 5818]
    alternate_net_sizes=[]
    for net_size in net_sizes:
        diffs = [abs(net_size - size) for size in sizes]
        min_index = diffs.index(min(diffs))
        alternate_net_sizes.append(sizes[min_index])
    
    size_to_power_dic = dict(list(zip(sizes,rel_powers)))

    alternate_net_sizes_powers = []
    for size in alternate_net_sizes:
        alternate_net_sizes_powers.append(size_to_power_dic[size])

    return alternate_net_sizes, alternate_net_sizes_powers

def get_k_backbones_subgraphs(net_name, phi, best_net_size_powers):
    f = open(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/{net_name}_{phi}_powers_nets.pkl', 'rb')
    x = pickle.load(f)
    f.close()
    x_dic = dict(x)
    subgraphs = []
    for net_size_power in best_net_size_powers:
        subgraphs.append(nx.from_dict_of_dicts(x_dic[net_size_power]))
    return subgraphs

#%%
if __name__ == "__main__":
    dist_net = nx.read_graphml('/Users/Bernardo/Desktop/thesis/frenchHSNet/um_frenchHSNet.gml')
    
    
    list_nr_edges = [326, 426, 500, 603, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4369, 4500, 5000, 5500, 5818]
    brs = backbone_random_subgraphs(dist_net, list_nr_edges, n=100)
    trs = threshold_random_subgraphs(dist_net, list_nr_edges, n=100)
    rrs = random_random_subgraphs(dist_net, list_nr_edges, n=100)
    tds = threshold_distortion_subgraphs(dist_net, list_nr_edges)
    tps = threshold_proximity_subgraphs(dist_net, list_nr_edges)

    brs_fcs = get_fraction_connected_subgraphs(brs)
    trs_fcs = get_fraction_connected_subgraphs(trs)
    rrs_fcs = get_fraction_connected_subgraphs(rrs)
    tds_fcs = get_fraction_connected_subgraphs(tds)
    tps_fcs = get_fraction_connected_subgraphs(tps)

    print(brs_fcs)
    print(trs_fcs)
    print(rrs_fcs)
    print(tds_fcs)
    print(tps_fcs)