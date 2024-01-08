# %% [markdown]
# # Inspecting Distance Backbones
#   
# ##### Author: M. Bernardo G. Pereira  
#   
# This work is inspired by the articles:  
#   
# *Brattig Correia R, Barrat A, Rocha LM (2023) Contact networks have small metric backbones that maintain community structure and are primary transmission subgraphs. PLoS Comput Biol 19(2): e1010854. https://doi.org/10.1371/journal.pcbi.1010854*  
#   
# *Costa, F.X., Correia, R.B., Rocha, L.M. (2023). The Distance Backbone of Directed Networks. In: Cherifi, H., Mantegna, R.N., Rocha, L.M., Cherifi, C., Micciche, S. (eds) Complex Networks and Their Applications XI. COMPLEX NETWORKS 2016 2022. Studies in Computational Intelligence, vol 1078. Springer, Cham. https://doi.org/10.1007/978-3-031-21131-7_11*

# %%
import sys
import os
import plot_networks as pn
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import time
import forceatlas2 as fa2
import random as rd

#%%
#net_name = 'frenchHSNet'
#net_name = 'exhibitNet'
net_name = 'workplaceNet'
#net_name = 'unitedstatesHSNet'
#net_name = 'medellinNet'
#net_name = 'manizalesNet'

#net_name = 'hospitalNet'
#net_name = 'conferenceNet'
#net_name = 'primaryschoolNet'
#net_name = 'undirUSairportsNet'
#net_name = ''


current_directory = os.getcwd()

child_directory = current_directory + f'/{net_name}'

#Network with just proximities
net = nx.read_graphml(child_directory+f'/{net_name}.gml')
# %%
#BACKBONES
#Network with just the ultrametric edges (UltraMetric Backbone)
umb_net = nx.read_graphml(child_directory+f'/umb_{net_name}.gml')

#Network with just the triangular edges (Euclidean Backbone)
eb_net = nx.read_graphml(child_directory+f'/eb_{net_name}.gml')

#Network with just the metric edges (Metric Backbone)
mb_net = nx.read_graphml(child_directory+f'/mb_{net_name}.gml')

#Network with just the triangular edges (Product Backbone)
pb_net = nx.read_graphml(child_directory+f'/pb_{net_name}.gml')

# %%
#COMPLETE NETWORKS WITH DISTORTIONS
#Network with proximities, distances and ultra-metric distortions
um_net = nx.read_graphml(child_directory+f'/um_{net_name}.gml')

#Network with proximities, distances and euclidean distortions
e_net = nx.read_graphml(child_directory+f'/e_{net_name}.gml')

#Network with proximities, distances and metric distortions
m_net = nx.read_graphml(child_directory+f'/m_{net_name}.gml')

#Network with proximities, distances and product distortions
p_net = nx.read_graphml(child_directory+f'/p_{net_name}.gml')
#%%
print(f'NET : {net_name}')

type_of_graph = str(type(net)).split('.')[-1][:-2]

#%%
#Deciding what kind of plot to make for the given network
#In the case of US airports networks the positions of given by their coordinates
if type_of_graph == 'DiGraph':
    if net_name=='dirUSairportsNet':
        df = pd.read_csv(current_directory+f'/{net_name}/T_MASTER_CORD.csv')
        keys = list(net.nodes())
        coords = [ (list(df[df['AIRPORT_ID']==int(node)]['LONGITUDE'])[-1], list(df[df['AIRPORT_ID']==int(node)]['LATITUDE'])[-1]) for node in keys]
        my_pos = dict(zip(keys, coords))
    else:
        my_pos=nx.spring_layout(net)

    largest = max(nx.strongly_connected_components(G), key=len)
    lscc = net.subgraph(largest)
    diam = nx.diameter(lscc)
    
    pn.plot_directed_net_with_colors(net, my_pos)
if type_of_graph == 'Graph':
    if net_name=='undirUSairportsNet':
        df = pd.read_csv(current_directory+f'/{net_name}/T_MASTER_CORD.csv')
        keys = list(net.nodes())
        coords = [ (list(df[df['AIRPORT_ID']==int(node)]['LONGITUDE'])[-1], list(df[df['AIRPORT_ID']==int(node)]['LATITUDE'])[-1]) for node in keys]
        my_pos = dict(zip(keys, coords))
    else: 
        init_pos = { i : (rd.random(), rd.random()) for i in m_net.nodes()} # Optionally specify positions as a dictionary 
        my_pos = fa2.forceatlas2_networkx_layout(m_net, init_pos, niter=1000) # Optionally specify iteration count 
    pn.plot_undirected_graph_with_colors(m_net, my_pos, my_node_size=20, node_labels_switch=False, edge_labels_switch = False)

    diam = nx.diameter(net)

#%%
nr_nodes = nx.number_of_nodes(net)
nr_edges = nx.number_of_edges(net)
avg_degree = np.mean([d for n, d in net.degree()])
density = nx.density(net)
avg_clustering = nx.average_clustering(net, weight='proximity')

umb_nr_edges = nx.number_of_edges(umb_net)
eb_nr_edges = nx.number_of_edges(eb_net)
mb_nr_edges = nx.number_of_edges(mb_net)
pb_nr_edges = nx.number_of_edges(pb_net)

print(f' Diameter : {diam}')
print(' ')
print(f' Average Degree : {avg_degree}')
print(' ')
print(f' Density : {density}')
print(' ')
print(f' Average Clustering Coefficient : {avg_clustering}')
print(' ')
print(f'Nr Nodes : {nr_nodes}')
print(' ')
print('Nr Edges:')
print(f' UltraMetric Backbone : {umb_nr_edges} | {round(umb_nr_edges*100/nr_edges,2)}%')
print(f' Euclidean Backbone : {eb_nr_edges} | {round(eb_nr_edges*100/nr_edges,2)}%')
print(f' Metric Backbone : {mb_nr_edges} | {round(mb_nr_edges*100/nr_edges,2)}%')
#print(f' Trigonometric Backbone : {tb_nr_edges} | {round(tb_nr_edges*100/nr_edges,2)}%')
print(f' Product Backbone : {pb_nr_edges} | {round(pb_nr_edges*100/nr_edges,2)}%')
print(f' Network : {nr_edges} | 100%')


#%%
degree_sequence = sorted((d for n, d in net.degree()), reverse=True)
um_degree_sequence = sorted((d for n, d in um_net.degree()), reverse=True)
e_degree_sequence = sorted((d for n, d in e_net.degree()), reverse=True)
m_degree_sequence = sorted((d for n, d in m_net.degree()), reverse=True)
p_degree_sequence = sorted((d for n, d in p_net.degree()), reverse=True)

clustering_coef_sequence = sorted(list(nx.clustering(net).values()), reverse=True)
um_clustering_coef_sequence = sorted(list(nx.clustering(umb_net).values()), reverse=True)
e_clustering_coef_sequence = sorted(list(nx.clustering(eb_net).values()), reverse=True)
m_clustering_coef_sequence = sorted(list(nx.clustering(mb_net).values()), reverse=True)
p_clustering_coef_sequence = sorted(list(nx.clustering(pb_net).values()), reverse=True)

proximity_sequence = sorted([d[f'proximity'] for (i,j,d) in net.edges(data=True)], reverse=True)

um_distance_sequence = sorted([d[f'distance'] for (i,j,d) in um_net.edges(data=True)], reverse=True)
e_distance_sequence = sorted([d[f'distance'] for (i,j,d) in e_net.edges(data=True)], reverse=True)
m_distance_sequence = sorted([d[f'distance'] for (i,j,d) in m_net.edges(data=True)], reverse=True)
p_distance_sequence = sorted([d[f'distance'] for (i,j,d) in p_net.edges(data=True)], reverse=True)

um_distortion_sequence = sorted([d[f'distortion'] for (i,j,d) in um_net.edges(data=True)], reverse=True)
e_distortion_sequence = sorted([d[f'distortion'] for (i,j,d) in e_net.edges(data=True)], reverse=True)
m_distortion_sequence = sorted([d[f'distortion'] for (i,j,d) in m_net.edges(data=True)], reverse=True)
p_distortion_sequence = sorted([d[f'distortion'] for (i,j,d) in p_net.edges(data=True)], reverse=True)

'''
#Nodes Attributes
pn.plot_attribute_sequence(net_name, 'Degree', 'Nodes', degree_sequence)
pn.plot_attribute_sequence(net_name+' UltraMetric Backbone', 'Degree', 'Nodes', um_degree_sequence)
pn.plot_attribute_sequence(net_name+' Euclidean Backbone', 'Degree', 'Nodes', e_degree_sequence)
pn.plot_attribute_sequence(net_name+' Metric Backbone', 'Degree', 'Nodes', m_degree_sequence)
pn.plot_attribute_sequence(net_name+' Product Backbone', 'Degree', 'Nodes', p_degree_sequence)

pn.plot_attribute_sequence(net_name, 'Clustering Coef', 'Nodes', clustering_coef_sequence)
pn.plot_attribute_sequence(net_name+' UltraMetric Backbone', 'Clustering Coef', 'Nodes', um_clustering_coef_sequence)
pn.plot_attribute_sequence(net_name+' Euclidean Backbone', 'Clustering Coef', 'Nodes', e_clustering_coef_sequence)
pn.plot_attribute_sequence(net_name+' Metric Backbone', 'Clustering Coef', 'Nodes', m_clustering_coef_sequence)
pn.plot_attribute_sequence(net_name+' Product Backbone', 'Clustering Coef', 'Nodes', p_clustering_coef_sequence)

#Edges Attributes
pn.plot_attribute_sequence(net_name, 'Proximity', 'Edges', proximity_sequence)

pn.plot_attribute_sequence(net_name+' UltraMetric Backbone', 'Distance', 'Edges', um_distance_sequence)
pn.plot_attribute_sequence(net_name+' Euclidean Backbone', 'Distance', 'Edges', e_distance_sequence)
pn.plot_attribute_sequence(net_name+' Metric Backbone', 'Distance', 'Edges', m_distance_sequence)
pn.plot_attribute_sequence(net_name+' Product Backbone', 'Distance', 'Edges', p_distance_sequence)

pn.plot_attribute_sequence(net_name+' UltraMetric Backbone', 'Distortion', 'Edges', um_distortion_sequence)
pn.plot_attribute_sequence(net_name+' Euclidean Backbone', 'Distortion', 'Edges', e_distortion_sequence)
pn.plot_attribute_sequence(net_name+' Metric Backbone', 'Distortion', 'Edges', m_distortion_sequence)
pn.plot_attribute_sequence(net_name+' Product Backbone', 'Distortion', 'Edges', p_distortion_sequence)
'''