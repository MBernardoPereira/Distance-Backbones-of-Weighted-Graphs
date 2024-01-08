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
import math
import time
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import distanceclosure as dc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import forceatlas2 as fa2
# %% [markdown]
# # Part 1 - Functions to Plot Graphs

# %%

def plot_undirected_net_with_colors(net):
    #pos = nx.spring_layout(net, seed=63)  # Seed layout for reproducibility
    pos = nx.circular_layout(net) 
    w_label = False
    colors = range(20) #np.arange(0.01,len(net.nodes)+0.01)
    options = {
        "node_color": "#A0CBE2",
        "edge_color": colors,
        "width": 1,
        "edge_cmap": plt.cm.Blues,
        "with_labels": w_label
    }
    nx.draw(net, pos, node_color="#A0CBE2", edge_color = colors, width= 1, edge_cmap = plt.cm.Blues, with_labels= False)
    plt.show()

def plot_undirected_net_with_colors2(net):
    #pos = nx.spring_layout(net, seed=63)  # Seed layout for reproducibility
    edges, weights = zip(*nx.get_edge_attributes(net,'distance').items())
    normalized_weights = list(weights / np.linalg.norm(weights))
    
    pos = nx.circular_layout(net) 
    nx.draw(net, pos, node_color='b', edgelist=edges, edge_color=normalized_weights, width=1, edge_cmap=plt.cm.Blues)
    plt.show()

def plot_undirected_graph_with_colors(net, pos, my_node_size=20, node_labels_switch=False, edge_labels_switch = False):
    #pos = nx.circular_layout(net)
    #pos = nx.planar_layout(net)
    #pos = nx.nx_pydot.graphviz_layout(net)
    
    edges, weights = zip(*nx.get_edge_attributes(net,'distance').items())
       
    nx.draw(net, pos, with_labels=node_labels_switch, 
                      font_weight='bold', 
                      node_size=my_node_size,
                      edgelist=edges,
                      edge_color=weights, 
                      #width=1,
                      edge_cmap = plt.cm.plasma
                      )
    
    if edge_labels_switch:
        edge_weight = nx.get_edge_attributes(net,'distance')
        nx.draw_networkx_edge_labels(net, pos, edge_labels = edge_weight)
    
    plt.show()

def plot_forceatlas2(net, my_node_size=20, node_labels_switch=False, edge_labels_switch = False):
    pos = { i : (rd.random(), rd.random()) for i in net.nodes()} # Optionally specify positions as a dictionary 

    pos = fa2.forceatlas2_networkx_layout(net, pos, niter=1000) # Optionally specify iteration count 
    
    nx.draw_networkx(net, pos, with_labels=node_labels_switch, font_weight='bold', node_size=my_node_size, width=0.5, edge_color='black')

    if edge_labels_switch:
        edge_weight = nx.get_edge_attributes(net,'distance')
        nx.draw_networkx_edge_labels(net, pos, edge_labels = edge_weight)

    #plt.savefig(f'{name_of_figure}.png', dpi=500, bbox_inches="tight")
    plt.show()

def plot_simple_weighted_graph(net, my_node_size=20, node_labels_switch=False, edge_labels_switch = False, att_name='distance'):
    pos = nx.circular_layout(net)
    #pos = nx.planar_layout(net)
    #pos = nx.nx_pydot.graphviz_layout(net)
    nx.draw(net, pos, with_labels=node_labels_switch, font_weight='bold', node_size=my_node_size)
    if edge_labels_switch:
        edge_weight = nx.get_edge_attributes(net,att_name)
        nx.draw_networkx_edge_labels(net, pos, edge_labels = edge_weight)
    plt.show()

def plot_directed_net_with_colors(net, pos):
    #I DONT KNOW WHATS GOING ON HERE, THIS IS A CODE EXAMPLE FROM MATPLOTLIB DOCS
    #pos = nx.spring_layout(net)
    #pos = nx.circular_layout(net)

    node_sizes = [0.0001 * i for i in range(len(net))]
    M = net.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(net, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        net,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=4,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=1,
    )
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()

# %%
def plot_net_degrees_distribution(net, name_of_net):
    type_of_net = str(type(net)).split('.')[-1][:-2]
    
    if type_of_net == 'DiGraph':
        #I DONT KNOW WHATS GOING ON HERE, THIS IS A CODE EXAMPLE FROM MATPLOTLIB DOCS
        
        in_degree_sequence = sorted((d for n, d in net.in_degree()), reverse=True)
        dmax = max(in_degree_sequence)

        fig = plt.figure("In-Degree Histogram", figsize=(4, 4))
        fig, ax = plt.subplots()
        ax.bar(*np.unique(in_degree_sequence, return_counts=True))
        ax.set_title("In-Degree Histogram "+name_of_net)
        ax.set_xlabel("In-Degree")
        ax.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()
        
        #-------------------------------
        
        out_degree_sequence = sorted((d for n, d in net.out_degree()), reverse=True)
        dmax = max(out_degree_sequence)

        fig = plt.figure("Out-Degree Histogram", figsize=(4, 4))
        fig, ax = plt.subplots()
        ax.bar(*np.unique(out_degree_sequence, return_counts=True))
        ax.set_title("Out-Degree Histogram "+name_of_net)
        ax.set_xlabel("Out-Degree")
        ax.set_ylabel("# of Nodes")
        
        fig.tight_layout()
        plt.show()
    
    else:
        #I DONT KNOW WHATS GOING ON HERE, THIS IS A CODE EXAMPLE FROM MATPLOTLIB DOCS
        
        degree_sequence = sorted((d for n, d in net.degree()), reverse=True)
        dmax = max(degree_sequence)

        #fig = plt.figure("Degree Histogram", figsize=(4, 4))
        #fig = plt.figure()
        
        fig, ax = plt.subplots()

        ax.bar(*np.unique(degree_sequence, return_counts=True))
        ax.set_title("Degree Histogram of "+name_of_net)
        ax.set_xlabel("Degree")
        ax.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()
        
# %%
def plot_net_proximities_distribution(net, name_of_net, x_lower_lim=0, x_upper_lim=1):
    
    proxs = [d['proximity'] for (i,j,d) in net.edges(data=True)]

    fig, ax = plt.subplots()
    ax.hist(proxs, bins=100, density=False, range=(x_lower_lim,x_upper_lim))
    ax.set_title("Edge Proximity Distribution of "+name_of_net)
    ax.set_xlabel("Proximity")
    ax.set_ylabel("# of Edges")

    fig.tight_layout()
    plt.show()

# %%
def plot_net_edge_attribute_distribution(net, name_of_net, edge_attribute, x_lower_lim=0, x_upper_lim=1):
    
    attrs = [d[f'{edge_attribute}'] for (i,j,d) in net.edges(data=True)]

    fig, ax = plt.subplots()

    if edge_attribute == 'proximity':
        x_lower_lim = 0
        x_upper_lim = x_upper_lim

    if edge_attribute == 'distance':
        x_lower_lim = 0
        x_upper_lim = max(attrs)

    ax.hist(attrs, bins='fd', density=False, range=(x_lower_lim,x_upper_lim))
    ax.set_title(f'Edge {edge_attribute} Distribution of {name_of_net}')
    ax.set_xlabel(f'{edge_attribute}')
    ax.set_ylabel("# of Edges")

    fig.tight_layout()
    plt.show()

def plot_attribute_sequence(net_name, attribute_name, graph_element, attribute_sequence):
    fig = plt.figure(figsize=(10, 5))  # Adjust the figure size

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(4, 4)

    # Adjust the space between the title and the rest of the figure with the 'y' parameter
    fig.suptitle(f"{net_name} | {attribute_name}", fontsize=16, y=0.8)  # Smaller 'y' value

    ax1 = fig.add_subplot(axgrid[1:, :2])  # Adjust the position of ax1
    ax1.plot(attribute_sequence, "r-", marker="o")
    ax1.set_title("Rank Plot")
    ax1.set_ylabel(f"{attribute_name}")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[1:, 2:])  # Adjust the position of ax2
    ax2.hist(attribute_sequence, bins='fd', density=False) #range=(min(attribute_sequence),max(attribute_sequence))
    ax2.set_title("Histogram")
    ax2.set_xlabel(f"{attribute_name}")
    ax2.set_ylabel(f"# of {graph_element}")

    fig.tight_layout()
    plt.show()




