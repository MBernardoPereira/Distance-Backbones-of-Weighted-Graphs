# %% [markdown]
# # Inspecting Distance Backbones
#   
# ##### Author: M. Bernardo G. Pereira  
#   
# This work follows on the articles:  
# 
# *Brattig Correia R, Barrat A, Rocha LM (2023) Contact networks have small metric backbones that maintain community structure and are primary transmission subgraphs. PLoS Comput Biol 19(2): e1010854. https://doi.org/10.1371/journal.pcbi.1010854*  
# *Simas T, Correia RB, Rocha LM. The distance backbone of complex networks. Journal of Complex Net- works. 2021; 9:cnab021. https://doi.org/10.1093/comnet/cnab021*  
# *Costa, F.X., Correia, R.B., Rocha, L.M. (2023). The Distance Backbone of Directed Networks. In: Cherifi, H., Mantegna, R.N., Rocha, L.M., Cherifi, C., Micciche, S. (eds) Complex Networks and Their Applications XI. COMPLEX NETWORKS 2016 2022. Studies in Computational Intelligence, vol 1078. Springer, Cham. https://doi.org/10.1007/978-3-031-21131-7_11*

# %%
import networkx as nx
import distanceclosure as dc
import numpy as np
import copy as cp
import os
# %%
#In proximity graphs weights are like probabilities, they range between [0,1]. The stronger (higher) the proximity, 
#the more the nodes are alike, or similar. These graphs are usually obtained from computing the pairwise similarity 
#among all pairs of variables (nodes).

#%%
def create_net_from_file(file, source_node_pos, target_node_pos, with_weight, weight_pos, with_node_labels, source_node_label_pos, target_node_label_pos):
    f = open(file)
    edgelist={}
    node_attributes = {}
   
    for line in f:
        #The strip is probably not doing anything
        line = line.strip().split()
        pair = (line[source_node_pos],line[target_node_pos])
        if with_node_labels == True:
            node_attributes[line[source_node_pos]] = line[source_node_label_pos]
            node_attributes[line[target_node_pos]] = line[target_node_label_pos]
        if with_weight==True:
            weight = int(line[weight_pos])
        else:
            weight = 1
        if pair not in edgelist.keys():
            edgelist[pair]=weight
        else:
            edgelist.update({pair: edgelist[pair]+weight})
    
    net = nx.from_edgelist(edgelist)
    nx.set_edge_attributes(net, name='contacts', values=edgelist)
    if with_node_labels == True:
        nx.set_node_attributes(net, node_attributes, name='group')
    
    #Adding the attribute proximity to the edges using the Jaccard measure
    #This is done by converting the weights to [0,1](proximity) using the Jacard measure and then converting to distance using phi
    
    #adding the attribute that keeps the sum of all the weights from edges coming out of each node
    for node in net.nodes:
        r = 0
        for neighbor in net.neighbors(node):
            r += net[node][neighbor]['contacts']
        net.nodes[node]['r'] = r
    
    for (u, v, d) in net.edges(data=True):
        #using jaccard coefficient on the weights of the edges
        r_ij = d['contacts']
        r_ii = net.nodes[u]['r']
        r_jj = net.nodes[v]['r']
        if (r_ii + r_jj - r_ij) >= 0:
            jc = r_ij / (r_ii + r_jj - r_ij)
        else:
            jc = 0.
        d['proximity'] = jc

    return net
#%%

#In distance graphs, weights are distances in a particular space (often not euclidean), 
#and they range between [0, infinity]. The smaller the distance, the closer together two nodes are in that space.

#hamacher produt t-norm
def phi_metric(p):
    return 1/p - 1

#product t-norm
def phi_product(p):
    return -np.log(p)

#trigonometric t-norm
def phi_trig(p):
    return -np.tan((np.pi/2)*(p-1))

#euclidean t-norm
def phi_euclidean(p):
    return (1/p - 1)**2

# %%
# Adding distance to the edges, based on their proximity, using a custom phi function
def add_distance_with_phi(net, phi):
    net = cp.deepcopy(net)
    for (u, v, d) in net.edges(data=True):
        prox = d['proximity']
        if prox==0:
            d['distance'] = np.inf     
        else:
            try:
                d['distance'] = phi(prox)
            except OverflowError as oe:
                print(f"OverflowError: {oe}")
                d['distance'] = np.inf
            except ZeroDivisionError as zde:
                print(f"ZeroDivisionError: {zde}")
                d['distance'] = np.inf
    return net

# %% [markdown]
# #####   Network with distortion data

#Assuming we have edges containing distance, we add the distortion and the proximity
def complete_edges_network(net, kind='metric'):
    type_of_net = str(type(net)).split('.')[-1][:-2]
    create_using = eval('nx.'+type_of_net)
    
    Dc = dc.distance_closure(net, kind=kind, weight='distance', only_backbone=True)
    Dc.remove_edges_from(nx.selfloop_edges(Dc))

    complete_edgelist = []
    for el in Dc.edges(data=True):
        dic = Dc[el[0]][el[1]]
        del dic['is_'+kind]
        #This formulation is to avoid division by 0 when metric_distance is 0
        if dic['distance'] == dic[kind+'_distance']:
            dic['distortion'] = 1.0
        else: 
            dic['distortion'] = dic['distance']/dic[kind+'_distance']
        complete_edgelist.append( (el[0], el[1], dic) )
        
    complete_net = create_using(complete_edgelist)
    
    return complete_net

# %% [markdown]
# #####  Distance Backbone

# Retrives the subgraph that has the triangular edges according to the distance closure
def distance_backbone_network(net):
    type_of_net = str(type(net)).split('.')[-1][:-2]
    create_using = eval('nx.'+type_of_net)

    Dc = dc.distance_closure(net, kind='metric', weight='distance', only_backbone=True)
    Dc.remove_edges_from(nx.selfloop_edges(Dc))
     
    triang_edges = [(i, j) for i, j, d in Dc.edges(data=True) if d['is_metric'] is True]
    triang_edgelist = []
    for el in triang_edges:
        dic = Dc[el[0]][el[1]]
        del dic['is_metric']
        triang_edgelist.append( (el[0], el[1], dic))
    b_net = create_using(triang_edgelist)
    return b_net

# %% [markdown]
# ##### UltraMetric Backbone

# Retrives the subgraph that has the ultrametric edges according to the ultrametric closure
def ultrametric_backbone_network(net):
    type_of_net = str(type(net)).split('.')[-1][:-2]
    create_using = eval('nx.'+type_of_net)

    Dcum = dc.distance_closure(net, kind='ultrametric', weight='distance', only_backbone=True)
    Dcum.remove_edges_from(nx.selfloop_edges(Dcum))

    ultrametric_edges = [(i, j) for i, j, d in Dcum.edges(data=True) if d['is_ultrametric'] is True]
    ultrametric_edgelist = []
    for el in ultrametric_edges:
        dic = Dcum[el[0]][el[1]]
        del dic['is_ultrametric']
        ultrametric_edgelist.append( (el[0], el[1], dic) )
        
    um_net = create_using(ultrametric_edgelist)
    return um_net
#%%

if __name__ == '__main__':

    dic = {'frenchHSNet': (1, 2, False, None, True, 3, 4), 
           'hospitalNet': (1, 2, False, None, True, 3, 4),
           'conferenceNet': (1, 2, False, None, False, None, None),
           'primaryschoolNet': (1, 2, False, None, True, 3, 4),
           'manizalesNet': (0, 1, False, None, False, None, None),
           'medellinNet': (0, 1, False, None, False, None, None),
           'unitedstatesHSNet': (0, 1, True, 2, False, None, None),
           'exhibitNet': (1, 2, False, None, False, None, None),
           'workplaceNet': (0, 1, True, 2, False, None, None)}
    
    #net_name = 'frenchHSNet'
    #net_name = 'hospitalNet'
    #net_name = 'conferenceNet'
    #net_name = 'primaryschoolNet'
    #net_name = 'manizalesNet'
    #net_name = 'medellinNet'
    #net_name = 'unitedstatesHSNet'
    #net_name = 'exhibitNet'
    net_name = 'workplaceNet'

    current_directory = os.getcwd()
    net_directory = current_directory + f'/{net_name}/'
    source_node_pos, target_node_pos, with_weight, weight_pos, with_node_labels, source_node_label_pos, target_node_label_pos = dic[net_name]
    print(dic[net_name])

    print('Producing Network with Proximities...')
    net = create_net_from_file(net_directory+f'{net_name}.txt', source_node_pos, target_node_pos, with_weight, weight_pos, with_node_labels, source_node_label_pos, target_node_label_pos)
    nx.write_graphml(net, net_directory+f'{net_name}.gml')

    print(f'NET : Nodes {nx.number_of_nodes(net)} | Edges {nx.number_of_edges(net)}')

    print('Producing Backbones...')
    net_metric = add_distance_with_phi(net, phi_metric)
    net_euclidean = add_distance_with_phi(net, phi_euclidean)
    net_trig = add_distance_with_phi(net, phi_trig)
    net_product = add_distance_with_phi(net, phi_product)

    umb_net = ultrametric_backbone_network(net_metric)
    print(f'UMB : Nodes {nx.number_of_nodes(umb_net)} | Edges {nx.number_of_edges(umb_net)} | Proportion {nx.number_of_edges(umb_net)*100/nx.number_of_edges(net)}%')
    eb_net = distance_backbone_network(net_euclidean)
    print(f'EB : Nodes {nx.number_of_nodes(eb_net)} | Edges {nx.number_of_edges(eb_net)} | Proportion {nx.number_of_edges(eb_net)*100/nx.number_of_edges(net)}%')
    mb_net = distance_backbone_network(net_metric)
    print(f'MB : Nodes {nx.number_of_nodes(mb_net)} | Edges {nx.number_of_edges(mb_net)} | Proportion {nx.number_of_edges(mb_net)*100/nx.number_of_edges(net)}%')
    tb_net = distance_backbone_network(net_trig)
    #print(f'TB : Nodes {nx.number_of_nodes(tb_net)} | Edges {nx.number_of_edges(tb_net)} | Proportion {nx.number_of_edges(tb_net)*100/nx.number_of_edges(net)}%')
    pb_net = distance_backbone_network(net_product)
    print(f'PB : Nodes {nx.number_of_nodes(pb_net)} | Edges {nx.number_of_edges(pb_net)} | Proportion {nx.number_of_edges(pb_net)*100/nx.number_of_edges(net)}%')

    nx.write_graphml(umb_net, net_directory+f'umb_{net_name}.gml')
    nx.write_graphml(mb_net, net_directory+f'mb_{net_name}.gml')
    nx.write_graphml(eb_net, net_directory+f'eb_{net_name}.gml')
    nx.write_graphml(tb_net, net_directory+f'tb_{net_name}.gml')
    nx.write_graphml(pb_net, net_directory+f'pb_{net_name}.gml')

    print('Producing Networks with Distortion Data...')
    um_net = complete_edges_network(net_metric, kind='ultrametric')
    m_net = complete_edges_network(net_metric, kind='metric')
    e_net = complete_edges_network(net_euclidean, kind='metric')
    t_net = complete_edges_network(net_trig, kind='metric')
    p_net = complete_edges_network(net_product, kind='metric')

    nx.write_graphml(um_net, net_directory+f'um_{net_name}.gml')
    nx.write_graphml(m_net, net_directory+f'm_{net_name}.gml')
    nx.write_graphml(e_net, net_directory+f'e_{net_name}.gml')
    nx.write_graphml(t_net, net_directory+f't_{net_name}.gml')
    nx.write_graphml(p_net, net_directory+f'p_{net_name}.gml')
