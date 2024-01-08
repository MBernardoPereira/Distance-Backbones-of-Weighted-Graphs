import multiprocessing as mp
from create_nets import distance_backbone_network, complete_edges_network, add_distance_with_phi
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import os
import sys
import time

dic = {'frenchHSNet':{'phi1': [10.3,13.5], 'phi2':[3,42], 'phi3':[11,15], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'undirUSairportsNet':{'phi1': [18,45], 'phi2':[4,60], 'phi3':[18,60], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'hospitalNet':{'phi1': [8,17], 'phi2':[2,60], 'phi3':[8,17], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'conferenceNet':{'phi1': [7,22], 'phi2':[2,60], 'phi3':[7,22], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'primaryschoolNet':{'phi1': [9,32], 'phi2':[2,60], 'phi3':[9,34], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'manizalesNet':{'phi1': [1,1], 'phi2':[1,1], 'phi3':[1,1], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }, 
       'medellinNet':{'phi1': [1,1], 'phi2':[1,1], 'phi3':[1,1], 'phi4':[80,80], 'phi5':[80,40], 'phi6':[80,40] }}

def get_edgelist_and_nr_backbone_edges(net_name, phi_name, k):
    phi_functions = {#Generators for Dombi T-Norms
                     'phi_D': lambda x: (1/x - 1)**k,
                     #Generators for Aczel-Alsina T-Norms
                     'phi_AA': lambda x: (-np.log(x))**k,
                     #Generators for Trigonometric T-Norms
                     'phi_T': lambda x: (-np.tan((np.pi/2)*(x-1)))**k,
                     #Generators for Frank T-Norms (The else case is because lim k->1 (k**x - 1)/(k-1) = x)
                     'phi_F': (lambda x: (-np.log( (k**x - 1)/(k-1) ))) if k != 1 else (lambda x: (-np.log(x))),
                     #Generators for Hamacher T-Norms
                     'phi_H': lambda x: (-np.log( x/(k + (1-k)*x))),
                     #Generators for Schweiser&Sklar4 T-Norms
                     'phi_SS4': lambda x: (1/(x**k) - 1),
                     #Generators for Trigonometric 2 T-Norms
                     'phi_T2': lambda x: (-np.tan((np.pi/2)*(x**k-1)))}
    
    phi = phi_functions.get(phi_name)
    if phi is None:
        raise ValueError("Invalid phi_name")

    print('k:', k)
    net_directory = os.getcwd() + f'/{net_name}'
    custom_net = nx.read_graphml(net_directory+f'/{net_name}.gml')
    custom_net = add_distance_with_phi(custom_net, phi)
    #custom_net = distance_backbone_network(custom_net)
    #nr_edges = nx.number_of_edges(custom_net)
    custom_net = complete_edges_network(custom_net, kind='metric')
    edgelist = list(custom_net.edges(data=True))
    edgelist = sorted([(edge[0],edge[1],edge[-1]['distortion']) for edge in edgelist], key= lambda x:x[-1])
    nr_backbone_edges = len([edge for edge in edgelist if edge[-1]==1])
    return edgelist, nr_backbone_edges

if __name__ == '__main__':
    ks_1 = [f'1/{n}' for n in np.arange(100,1,-1)]+[f'{int(n)}' for n in np.arange(1,100.5,1)]
    ks_05 = [f'1/{n}' for n in np.arange(99.5,0.5,-1)]+[f'{n}' for n in np.arange(1.5,100.5,1)]
    ks_025 = [f'1/{n}' for n in np.arange(100-0.25,-0.25,-0.5)]+[f'{n}' for n in np.arange(1.25,100.25,0.5)]
    #ks_0125 = [f'1/{n}' for n in np.arange(100-0.125,-0.125,-(0.125*2))]+[f'{n}' for n in np.arange(1+0.125,100+0.125,2*0.125)]

    ks_labels = sorted(ks_025, key= lambda x:eval(x))
    ks_values = sorted([eval(k) for k in ks_labels])

    #mid_index = round(len(ks_labels)/2)
    #ks_labels = ks_labels[mid_index-3:mid_index+3]
    #ks_values = ks_values[mid_index-3:mid_index+3]

    #net_name = 'frenchHSNet'
    #net_name = 'undirUSairportsNet'
    #net_name = 'hospitalNet'
    #net_name = 'conferenceNet'
    #net_name = 'primaryschoolNet'
    #net_name = 'exhibitNet'
    #net_name = 'unitedstatesHSNet'
    #net_name = 'workplaceNet'
    
    for net_name in ['frenchHSNet', 'exhibitNet', 'workplaceNet']:
        net_directory = os.getcwd()+f'/{net_name}/'
        print(net_name)
        t0 = time.time()
        for phi_name in ['phi_D', 'phi_AA', 'phi_F', 'phi_H', 'phi_SS4']:
            print(phi_name)

            n_cpu = mp.cpu_count()
            pool = mp.Pool(processes=n_cpu)
            print('starting multiprocessing...')

            net_phi_k = [ (net_name, phi_name, k) for k in ks_values]
            data = pool.starmap(get_edgelist_and_nr_backbone_edges, net_phi_k)
            edgelist_data, backbone_size_data = zip(*data)
            
            edgelist_data_df = pd.DataFrame(list(edgelist_data), index=ks_labels).T
            edgelist_data_df.to_csv(net_directory+f'/backbones_sizes/{phi_name}_backbones_edgelists_025.csv', index=False)
            edgelist_data_df.to_html(net_directory+f'/backbones_sizes/{phi_name}_backbones_edgelists_025.html', index=False)

            backbones_sizes_df = pd.DataFrame([list(backbone_size_data)], columns=ks_labels, index=['size'])
            backbones_sizes_df.to_csv(net_directory+f'/backbones_sizes/{phi_name}_backbones_sizes_025.csv', index=False)
            backbones_sizes_df.to_html(net_directory+f'/backbones_sizes/{phi_name}_backbones_sizes_025.html', index=False)
            print(' ')
        
        t1 = time.time()
        print(f'Time: {round((t1-t0)//60)}m {round((t1-t0)%60)}s')
