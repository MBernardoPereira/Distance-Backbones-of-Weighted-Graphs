#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import si_functions as sif
import subgraphs_functions as sgf
import networkx as nx
import multiprocessing as mp
import numpy as np
import time
import random as rd

#%%
#sparsifier_name is one of ['threshold_subgraph', 'random_subgraph', 'ultrametric_backbone', 'euclidean_backbone', 'metric_backbone', 'product_backbone'] 
#type_of_subgraphs is one of ['random','distortion']

def get_random_seed_nodes(net, p=0.1):
    sampling_universe=list(net.nodes)
    sample_size = round(p*len(sampling_universe))
    seeds = rd.sample(sampling_universe, sample_size)
    return seeds

def read_subgraphs_sizes(net_name):
    """
    Documentation
    """
    print(f'Reading subgraphs sizes | Network {net_name} ...')
    current_directory = os.getcwd()
    net_directory = current_directory +f'/{net_name}'
    f = open(net_directory+f'/subgraphs_sizes.txt', 'r')
    subgraphs_sizes_by_backbone = {}
    for line in f.readlines():
        pair = line.split(':')
        subgraphs_sizes_by_backbone[pair[0]] = eval(pair[1].strip())
    f.close()

    subgraphs_sizes_percentages_by_backbone = {}
    for key in subgraphs_sizes_by_backbone.keys():
        sizes = subgraphs_sizes_by_backbone[key]
        max_size = sizes[-1]
        percentages = [ round((s*100/max_size),2) for s in sizes]
        subgraphs_sizes_percentages_by_backbone[key]=percentages
    return subgraphs_sizes_by_backbone, subgraphs_sizes_percentages_by_backbone

def produce_info_file(net_name, net_directory, sparsifier_name, type_of_subgraphs, n_real, n_sims, list_net_sizes, list_percentages, seed_nodes, appendix, fraction_connected_subgraphs=None):
    """
    Documentation
    """
    print(f'Producing info file | Network {net_name} | Sparsifier {sparsifier_name} |  Subgraphs {type_of_subgraphs} ...')
    f = open(net_directory+f'/si/info_{net_name}_{sparsifier_name}_{type_of_subgraphs}{appendix}.txt', 'w')
    f.write('network:'+net_name+'\n')
    f.write('ep_model:si\n')
    f.write('sparsifier_name:'+sparsifier_name+'\n')
    f.write('type_of_subgraphs:'+type_of_subgraphs+'\n')
    f.write('list_net_sizes:'+str(list_net_sizes)+'\n')
    f.write('list_net_sizes_percentages:'+str(list_percentages)+'\n')
    f.write('seed_nodes:'+str(seed_nodes)+'\n')
    f.write('nr_simulations:'+str(n_sims)+'\n')
    f.write('nr_realizations:'+str(n_real)+'\n')   
    f.write('fcs:'+str(fraction_connected_subgraphs))  
    f.close()

def produce_data_file(net_name, net_directory, sparsifier_name, type_of_subgraphs, subgraphs, seed_nodes, nr_simulations, appendix):
    """
    Documentation
    """

    print(f'Extracting data from | Network {net_name} | Sparsifier {sparsifier_name} |  Subgraphs {type_of_subgraphs} ...')
    t0 = time.time()
    n_cpu = mp.cpu_count()

    subgraphs_data = []

    print('shape of subgraphs array:',np.shape(subgraphs))
    #Convert subgraphs from shape nr_net_realizations x nr_net_sizes to shape nr_net_sizes x nr_net_realizations
    subgraphs = list(zip(*subgraphs))
    subgraphs = [list(subgraph) for subgraph in subgraphs]
    print('shape of subgraphs array:',np.shape(subgraphs))

    #Number of different network sizes
    nr_net_sizes = len(subgraphs)

    for nnr in range(nr_net_sizes):
        current_subgraphs = subgraphs[nnr]
        nr_current_subgraphs = len(current_subgraphs)
        pool = mp.Pool(processes=n_cpu) #n_cpu-1

        get_data_function = sif.get_data_from_subgraph
        #sif.get_data_from_subgraph(subgraph, seed_nodes, number_of_spreadings, data_retrieved)
        inputs = zip(current_subgraphs, 
                    [seed_nodes for i in range(nr_current_subgraphs)], 
                    [nr_simulations for i in range(nr_current_subgraphs)], 
                    ['t_values' for i in range(nr_current_subgraphs)])
        
        current_subgraphs_data = pool.starmap(get_data_function, inputs)
        pool.close()
        subgraphs_data.append(current_subgraphs_data)
        #print('current_subgraphs_data:',current_subgraphs_data,'\n')
    
    #Writing data to txt file   
    f = open(net_directory+f'/si/data_{net_name}_{sparsifier_name}_{type_of_subgraphs}{appendix}.txt', 'w')
    f.write(str(subgraphs_data))
    f.close()

    t1 = time.time()
    print(f'Execution time {(t1 - t0)//60}m{round((t1 - t0)%60)}s')

# %%        
if __name__ == '__main__':   
    t00 = time.time()

    #Appendix to the name of the files with a unique code to identify different runs
    appendix=f'_{hex(int(time.time()))}'

    #net_name = 'frenchHSNet'
    #net_name = 'exhibitNet' 
    net_name = 'workplaceNet'
    #net_name = 'unitedstatesHSNet' 
    #net_name = 'medellinNet'
    #net_name = 'manizalesNet'

    #net_name = 'primaryschoolNet' 
    #net_name = 'hospitalNet' 
    #net_name = 'conferenceNet' 

    current_directory = os.getcwd()
    net_directory = current_directory +f'/{net_name}'

    #Number of (random) subgraphs realizations for each value of percentage
    n_real=100

    #Number of SI spreading realizations for each subgraph
    n_sims=10

    #Create the network
    net = nx.read_graphml(net_directory+f'/{net_name}.gml')

    #Fix the set of seed nodes
    #seed_nodes = get_random_seed_nodes(net, p=3/len(net.nodes))
    seed_nodes = get_random_seed_nodes(net, p=0.1)
    print('nr_seed_nodes:',len(seed_nodes))

    #Load subgraphs sizes for each backbone
    subgraphs_sizes_by_backbone, subgraphs_sizes_percentages_by_backbone = read_subgraphs_sizes(net_name)
    umb_subgraphs_sizes, umb_subgraphs_sizes_percentages = subgraphs_sizes_by_backbone['um'], subgraphs_sizes_percentages_by_backbone['um']
    eb_subgraphs_sizes, eb_subgraphs_sizes_percentages = subgraphs_sizes_by_backbone['e'], subgraphs_sizes_percentages_by_backbone['e']
    mb_subgraphs_sizes, mb_subgraphs_sizes_percentages = subgraphs_sizes_by_backbone['m'], subgraphs_sizes_percentages_by_backbone['m']
    pb_subgraphs_sizes, pb_subgraphs_sizes_percentages = subgraphs_sizes_by_backbone['p'], subgraphs_sizes_percentages_by_backbone['p']

    #Load distance networks with ULTRAMETRIC, EUCLIDEAN, METRIC and PRODUCT distortions
    um_net = nx.read_graphml(net_directory+f'/um_{net_name}.gml')
    e_net = nx.read_graphml(net_directory+f'/e_{net_name}.gml')
    m_net = nx.read_graphml(net_directory+f'/m_{net_name}.gml')
    p_net = nx.read_graphml(net_directory+f'/p_{net_name}.gml')

    #THRESHOLD PROXIMITY RANDOM SUBGRAPHS
    tr_subgraphs = sgf.threshold_random_subgraphs(p_net, pb_subgraphs_sizes, n=n_real)
    trs_fcs = sgf.get_fraction_connected_subgraphs(tr_subgraphs)
    produce_info_file(net_name, net_directory, 'threshold_proximity', 'random', n_real, n_sims, pb_subgraphs_sizes, pb_subgraphs_sizes_percentages, seed_nodes, '_pb'+appendix, fraction_connected_subgraphs=trs_fcs)
    produce_data_file(net_name, net_directory, 'threshold_proximity', 'random', tr_subgraphs, seed_nodes, n_sims, '_pb'+appendix)

    #THRESHOLD PROXIMITY RANDOM SUBGRAPHS
    tr_subgraphs = sgf.threshold_random_subgraphs(m_net, mb_subgraphs_sizes, n=n_real)
    trs_fcs = sgf.get_fraction_connected_subgraphs(tr_subgraphs)
    produce_info_file(net_name, net_directory, 'threshold_proximity', 'random', n_real, n_sims, mb_subgraphs_sizes, mb_subgraphs_sizes_percentages, seed_nodes, '_mb'+appendix, fraction_connected_subgraphs=trs_fcs)
    produce_data_file(net_name, net_directory, 'threshold_proximity', 'random', tr_subgraphs, seed_nodes, n_sims, '_mb'+appendix)

    #THRESHOLD PROXIMITY RANDOM SUBGRAPHS
    tr_subgraphs = sgf.threshold_random_subgraphs(e_net, eb_subgraphs_sizes, n=n_real)
    trs_fcs = sgf.get_fraction_connected_subgraphs(tr_subgraphs)
    produce_info_file(net_name, net_directory, 'threshold_proximity', 'random', n_real, n_sims, eb_subgraphs_sizes, eb_subgraphs_sizes_percentages, seed_nodes, '_eb'+appendix, fraction_connected_subgraphs=trs_fcs)
    produce_data_file(net_name, net_directory, 'threshold_proximity', 'random', tr_subgraphs, seed_nodes, n_sims, '_eb'+appendix)

    #THRESHOLD PROXIMITY RANDOM SUBGRAPHS
    tr_subgraphs = sgf.threshold_random_subgraphs(um_net, umb_subgraphs_sizes, n=n_real)
    trs_fcs = sgf.get_fraction_connected_subgraphs(tr_subgraphs)
    produce_info_file(net_name, net_directory, 'threshold_proximity', 'random', n_real, n_sims, umb_subgraphs_sizes, umb_subgraphs_sizes_percentages, seed_nodes, '_umb'+appendix, fraction_connected_subgraphs=trs_fcs)
    produce_data_file(net_name, net_directory, 'threshold_proximity', 'random', tr_subgraphs, seed_nodes, n_sims, '_umb'+appendix)

    #RANDOM SUBGRAPH RANDOM SUBGRAPHS
    rr_subgraphs = sgf.random_random_subgraphs(um_net, umb_subgraphs_sizes, n=n_real)
    rrs_fcs = sgf.get_fraction_connected_subgraphs(rr_subgraphs)
    produce_info_file(net_name, net_directory, 'random_subgraph', 'random', n_real, n_sims, umb_subgraphs_sizes, umb_subgraphs_sizes_percentages, seed_nodes, appendix, fraction_connected_subgraphs=rrs_fcs)
    produce_data_file(net_name, net_directory, 'random_subgraph', 'random', rr_subgraphs, seed_nodes, n_sims, appendix)

    #THRESHOLD PROXIMITY THRESHOLD SUBGRAPHS
    tp_subgraphs = sgf.threshold_proximity_subgraphs(um_net, umb_subgraphs_sizes)
    tps_fcs = sgf.get_fraction_connected_subgraphs(tp_subgraphs)
    produce_info_file(net_name, net_directory, 'threshold_proximity', 'threshold', n_real, n_sims, umb_subgraphs_sizes, umb_subgraphs_sizes_percentages, seed_nodes, appendix, fraction_connected_subgraphs=tps_fcs)
    produce_data_file(net_name, net_directory,'threshold_proximity', 'threshold', tp_subgraphs, seed_nodes, n_sims, appendix)

    #ULTRAMETRIC BACKBONE RANDOM SUBGRAPHS
    umbr_subgraphs = sgf.backbone_random_subgraphs(um_net, umb_subgraphs_sizes, n=n_real)
    produce_info_file(net_name, net_directory, 'ultrametric_backbone', 'random', n_real, n_sims, umb_subgraphs_sizes, umb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'ultrametric_backbone', 'random', umbr_subgraphs, seed_nodes, n_sims, appendix)

    #ULTRAMETRIC BACKBONE DISTORTION SUBGRAPHS
    umbtd_subgraphs = sgf.threshold_distortion_subgraphs(um_net, umb_subgraphs_sizes)
    produce_info_file(net_name, net_directory, 'ultrametric_backbone', 'distortion', 1, n_sims, umb_subgraphs_sizes, umb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'ultrametric_backbone', 'distortion', umbtd_subgraphs, seed_nodes, n_sims, appendix)

    #EUCLIDEAN BACKBONE RANDOM SUBGRAPHS
    ebr_subgraphs = sgf.backbone_random_subgraphs(e_net, eb_subgraphs_sizes, n=n_real)
    produce_info_file(net_name, net_directory, 'euclidean_backbone', 'random', n_real, n_sims, eb_subgraphs_sizes, eb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'euclidean_backbone', 'random', ebr_subgraphs, seed_nodes, n_sims, appendix)

    #EUCLIDEAN BACKBONE DISTORTION SUBGRAPHS
    ebtd_subgraphs = sgf.threshold_distortion_subgraphs(e_net, eb_subgraphs_sizes)
    produce_info_file(net_name, net_directory, 'euclidean_backbone', 'distortion', 1, n_sims, eb_subgraphs_sizes, eb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'euclidean_backbone', 'distortion', ebtd_subgraphs, seed_nodes, n_sims, appendix)

    #METRIC BACKBONE RANDOM SUBGRAPHS
    mbr_subgraphs = sgf.backbone_random_subgraphs(m_net, mb_subgraphs_sizes, n=n_real)
    produce_info_file(net_name, net_directory, 'metric_backbone', 'random', n_real, n_sims, mb_subgraphs_sizes, mb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'metric_backbone', 'random', mbr_subgraphs, seed_nodes, n_sims, appendix)

    #METRIC BACKBONE DISTORTION SUBGRAPHS
    mbtd_subgraphs = sgf.threshold_distortion_subgraphs(m_net, mb_subgraphs_sizes)
    produce_info_file(net_name, net_directory, 'metric_backbone', 'distortion', 1, n_sims, mb_subgraphs_sizes, mb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'metric_backbone', 'distortion', mbtd_subgraphs, seed_nodes, n_sims, appendix)

    #PRODUCT BACKBONE RANDOM SUBGRAPHS
    pbr_subgraphs = sgf.backbone_random_subgraphs(p_net, pb_subgraphs_sizes, n=n_real)
    produce_info_file(net_name, net_directory, 'product_backbone', 'random', n_real, n_sims, pb_subgraphs_sizes, pb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'product_backbone', 'random', pbr_subgraphs, seed_nodes, n_sims, appendix)

    #PRODUCT BACKBONE DISTORTION SUBGRAPHS
    pbtd_subgraphs = sgf.threshold_distortion_subgraphs(p_net, pb_subgraphs_sizes)
    produce_info_file(net_name, net_directory, 'product_backbone', 'distortion', 1, n_sims, pb_subgraphs_sizes, pb_subgraphs_sizes_percentages, seed_nodes, appendix)
    produce_data_file(net_name, net_directory, 'product_backbone', 'distortion', pbtd_subgraphs, seed_nodes, n_sims, appendix)

    t01 = time.time()
    print(f'Execution time {(t01 - t00)//60}m{round((t01 - t00)%60)}s')
       