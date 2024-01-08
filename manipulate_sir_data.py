#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

#info_frenchHSNet_euclidean_backbone_distortion_0x652b4bfb.txt
net_name_to_info_code = {
                         'frenchHSNet': {3:'0x65371246', 4:'0x65379c87', 5:'0x6537af98'},
                         'exhibitNet': {3:'0x653774e8', 4:'0x653779c9', 5:'0x65377edf'},
                         'workplaceNet': {3:'0x653784b4', 4:'0x65378a7a', 5:'0x653791d0'}
                         }

#Using this function we compute the data to plot and save it in a text file
def save_plot_data_to_file(net_name, sparsifier_name, subgraphs_type, info_code):
    #Import data from data file
    current_directory = os.getcwd()
    f = open(current_directory+f'/{net_name}/sir/data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'r')
    original_data = f.read()
    original_data = original_data.replace('nan', 'np.nan')
    original_data=eval(original_data)
    f.close()

    n0,n1,n2,n3 = np.shape(original_data)
    #Computing metrics for each realization/seed
    means = np.nanmean(original_data, axis=-1)
    stds = np.nanstd(original_data, ddof=0, axis=-1)

    means_agg, stds_agg = [], []
    for i in range(n0):
        mean_agg = np.nanmean(means[i,:,:])
        std_agg = np.sqrt(np.nansum(stds[i,:,:])/(n1*n2))
        means_agg.append(mean_agg)
        stds_agg.append(std_agg)

    f = open(current_directory+f'/{net_name}/sir/plot_data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'w')
    f.write(f'{means_agg}\n')
    f.write(f'{stds_agg}\n')
    f.close()
    return 

#%%
if __name__ == '__main__':
    
    net_name = 'frenchHSNet'
    beta=5
    #net_name = 'exhibitNet'
    #net_name = 'workplaceNet'
    #net_name = 'medellinNet'
    #net_name = 'manizalesNet'
    
    save_plot_data_to_file(net_name, 'threshold_proximity', 'threshold', net_name_to_info_code[net_name][beta])

    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'umb_'+net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'eb_'+net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'mb_'+net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'pb_'+net_name_to_info_code[net_name][beta])

    save_plot_data_to_file(net_name, 'random_subgraph', 'random', net_name_to_info_code[net_name][beta])

    save_plot_data_to_file(net_name, 'ultrametric_backbone', 'random', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'euclidean_backbone', 'random', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'metric_backbone', 'random', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'product_backbone', 'random', net_name_to_info_code[net_name][beta])
    
    save_plot_data_to_file(net_name, 'ultrametric_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'euclidean_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'metric_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    save_plot_data_to_file(net_name, 'product_backbone', 'distortion', net_name_to_info_code[net_name][beta])


    