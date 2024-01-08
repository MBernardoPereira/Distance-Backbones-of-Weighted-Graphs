#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

#info_frenchHSNet_euclidean_backbone_distortion_0x652b4bfb.txt
net_name_to_info_code = {'frenchHSNet': '0x652b4bfb',
                         'exhibitNet': '0x652b4bfb',
                         'unitedstatesHSNet': '0x652b4bfb',
                         'medellinNet': '0x652b4bfb',
                         'manizalesNet': '0x652b4bfb'}

def get_t_data_with_error_propagation(ts):
    #Shape of ts: (nr_net_sizes, nr_realizations, nr_seed_nodes, nr_simulations)

    #Computing metrics for each realization/seed
    means = np.nanmean(ts, axis=-1)
    stds = np.nanstd(ts, ddof=0, axis=-1)

    #Computing proportion metrics for each realization/seed
    n0,n1,n2,n3 = np.shape(ts)
    ts_prop_metrics = np.empty((n0,n1,n2,2))
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                mean_prop = means[i,j,k]/means[-1,j,k]
                var_prop = mean_prop**2 * ( (stds[i,j,k]**2/means[i,j,k]**2) + (stds[-1,j,k]**2/means[-1,j,k]**2) )
                ts_prop_metrics[i,j,k] = [mean_prop, var_prop]           

    #Computing proportion metrics for each network size by aggregating/averaging over nr_realizations and nr_seed_nodes
    ts_agg_prop_metrics = np.empty((n0,2))
    for i in range(n0):
        mean_prop_agg = np.nanmean(ts_prop_metrics[i,:,:,0])
        std_prop_agg = np.sqrt(np.nansum(ts_prop_metrics[i,:,:,1])/(n1*n2))
        ts_agg_prop_metrics[i] = [mean_prop_agg, std_prop_agg]
            
    means, stds = [list(el) for el in list(zip(*ts_agg_prop_metrics))]
    return means, stds

#Using this function we only compute the data to plot
def get_proportions_data_to_plot(net_name, sparsifier_name, subgraphs_type, info_code):
    info_code = net_name_to_info_code[net_name]
    #Import data from data file
    current_directory = os.getcwd()
    f = open(current_directory+f'/{net_name}/si/data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'r')
    original_data = f.read()
    original_data = original_data.replace('nan', 'np.nan')
    original_data=eval(original_data)
    f.close()
    data_split_by_ts = np.moveaxis(original_data, 2, 0)
    #Shape: (2, nr_net_sizes, nr_realizations, nr_seed_nodes, nr_simulations), where the 2 comes from t_halfs and t_alls
    
    t_halfs, t_alls = data_split_by_ts
    means_t_half, stds_t_half = get_t_data_with_error_propagation(t_halfs)
    means_t_all, stds_t_all = get_t_data_with_error_propagation(t_alls)
    return means_t_half, stds_t_half, means_t_all, stds_t_all

#Using this function we compute the data to plot and save it in a text file
def save_plot_data_to_file(net_name, sparsifier_name, subgraphs_type, info_code):
    #Import data from data file
    current_directory = os.getcwd()
    f = open(current_directory+f'/{net_name}/si/data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'r')
    original_data = f.read()
    original_data = original_data.replace('nan', 'np.nan')
    original_data=eval(original_data)
    f.close()
    data_split_by_ts = np.moveaxis(original_data, 2, 0)
    #Shape: (2, nr_net_sizes, nr_realizations, nr_seed_nodes, nr_simulations), where the 2 comes from t_halfs and t_alls
    
    t_halfs, t_alls = data_split_by_ts
    means_t_half, stds_t_half = get_t_data_with_error_propagation(t_halfs)
    means_t_all, stds_t_all = get_t_data_with_error_propagation(t_alls)

    f = open(current_directory+f'/{net_name}/si/plot_data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'w')
    f.write(f'{means_t_half}\n')
    f.write(f'{stds_t_half}\n')
    f.write(f'{means_t_all}\n')
    f.write(f'{stds_t_all}\n')
    f.close()
    return 

#%%
if __name__ == '__main__':
    ##############################################################################################################
    #MAKE THE PLOTS SUCH THAT FOR PLOTS THAT ONLY HAVE ONE BACKBONE, I REMOVE ALL THE DATA FOR THE OTHER BACKBONES
    ##############################################################################################################
    
    net_name = 'frenchHSNet'
    #net_name = 'exhibitNet'
    #net_name = 'workplaceNet'
    #net_name = 'medellinNet'
    #net_name = 'manizalesNet'
    
    save_plot_data_to_file(net_name, 'threshold_proximity', 'threshold', net_name_to_info_code[net_name])

    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'umb_'+net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'eb_'+net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'mb_'+net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'threshold_proximity', 'random', 'pb_'+net_name_to_info_code[net_name])

    save_plot_data_to_file(net_name, 'random_subgraph', 'random', net_name_to_info_code[net_name])

    save_plot_data_to_file(net_name, 'ultrametric_backbone', 'random', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'euclidean_backbone', 'random', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'metric_backbone', 'random', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'product_backbone', 'random', net_name_to_info_code[net_name])
    
    save_plot_data_to_file(net_name, 'ultrametric_backbone', 'distortion', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'euclidean_backbone', 'distortion', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'metric_backbone', 'distortion', net_name_to_info_code[net_name])
    save_plot_data_to_file(net_name, 'product_backbone', 'distortion', net_name_to_info_code[net_name])


    