#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
import pandas as pd

net_name_to_info_code = {
                         'frenchHSNet': {3:'0x65371246', 4:'0x65379c87', 5:'0x6537af98'},
                         'exhibitNet': {3:'0x653774e8', 4:'0x653779c9', 5:'0x65377edf'},
                         'workplaceNet': {3:'0x653784b4', 4:'0x65378a7a', 5:'0x653791d0'}
                         }

def get_info_from_plot_name(net_name, sparsifier_name, subgraphs_type, info_code):
    current_directory = os.getcwd()
    #Get backbones sizes from the subgraphs_sizes.txt file
    f = open(current_directory+f'/{net_name}/subgraphs_sizes.txt', 'r')
    info_lines = f.readlines()
    f.close()
    info_dic = {}
    for line in info_lines:
        info_pair = line.strip().split(':')
        info_dic[info_pair[0]] = eval(info_pair[1])[0]
        
    #Get all the other information in the info file
    f = open(current_directory+f'/{net_name}/sir/info_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'r')
    info_lines = f.readlines()
    f.close()
    for line in info_lines:
        info_pair = line.strip().split(':')
        try:
            info_dic[info_pair[0]] = eval(info_pair[1])
        except:
            info_dic[info_pair[0]] = info_pair[1]

    #Information available in info_dic:
    #'network', 'ep_model', 'sparsifier_name', 'type_of_subgraphs', 'beta'
    #'list_net_sizes', 'list_net_sizes_percentages', 'seed_nodes', 'nr_simulations', 'nr_realizations' 'fcs'
    #'um', 'e', 'm', 'p' 
    return info_dic

def get_data_plot_from_files(net_name, sparsifier_name, subgraphs_type, info_code):
    #Import data from data file
    current_directory = os.getcwd()
    f = open(current_directory+f'/{net_name}/sir/plot_data_{net_name}_{sparsifier_name}_{subgraphs_type}_{info_code}.txt', 'r')
    lines = f.readlines()
    means = eval(lines[0].replace('nan', 'np.nan'))
    stds = eval(lines[1].replace('nan', 'np.nan'))
    return means, stds

#%%
def get_labels_from_ticks(ticks_values, largest_size):
    #Receives the list of ticks with number of edges and a list of ticks with percentages sizes
    #returns a list o labels for the x-axis where the first has the notation for the size of backbone and the last has the notation for the whole network
    labels = []
    for i in range(len(ticks_values)):
        percentage = round(ticks_values[i]*100/largest_size,1)
        if i==0:
            labels.append(f'{ticks_values[i]}\n {percentage}%\n'+r'$|\{b_{ij}\}|$')
        elif i==len(ticks_values)-1:
            labels.append(f'{ticks_values[i]}\n {percentage}%\n'+r'$|\{d_{ij}\}|$')
        else:
            labels.append(f'{ticks_values[i]}\n {percentage}%')
    return labels

def get_ticks_with_min_spacing(ticks, min_spacing=500):
    best_ticks = [ticks[0]]
    for i in range(1,len(ticks)-2):
        if ticks[i]-best_ticks[-1]>=min_spacing:
            best_ticks.append(ticks[i])
    if ticks[-1]-ticks[-2]>=500 and ticks[-2]-ticks[-3]>=min_spacing:
        best_ticks.append(ticks[-2])
    best_ticks.append(ticks[-1])
    return best_ticks

def get_ticks_without_other_backbones_sizes(ticks):
    best_ticks = [ticks[0]]
    for i in range(1,len(ticks)-1):
        if ticks[i]%500==0:
            best_ticks.append(ticks[i])
    best_ticks.append(ticks[-1])
    return best_ticks

def get_backbone_dataframes_from_plot_data(net_name, net_size, sparsifier_name, subgraphs_type, info_code):
    means, stds = get_data_plot_from_files(net_name, sparsifier_name, subgraphs_type, info_code)
    info = get_info_from_plot_name(net_name, sparsifier_name, subgraphs_type, info_code)
    list_net_sizes = info['list_net_sizes']
    list_net_sizes_labels = get_labels_from_ticks(list_net_sizes, net_size)
    complete_df = pd.DataFrame([list_net_sizes_labels, means, stds], columns=list_net_sizes, index=['label', 'r_inf_means', 'r_inf_stds'])
    list_net_sizes_without_other_backbones = get_ticks_without_other_backbones_sizes(list_net_sizes)
    reduced_df = complete_df[list_net_sizes_without_other_backbones]
    return complete_df, reduced_df

def get_threshold_dataframe_from_plot_data(net_name, net_size, sparsifier_name, subgraphs_type, info_code):
    means, stds = get_data_plot_from_files(net_name, sparsifier_name, subgraphs_type, info_code)
    info = get_info_from_plot_name(net_name, sparsifier_name, subgraphs_type, info_code)
    list_net_sizes = info['list_net_sizes']
    list_net_sizes_labels = get_labels_from_ticks(list_net_sizes, net_size)
    fcs = info['fcs']

    complete_df = pd.DataFrame([list_net_sizes_labels, means, stds, fcs], columns=list_net_sizes, index=['label', 'r_inf_means', 'r_inf_stds', 'fcs'])

    list_net_sizes_without_other_backbones = get_ticks_without_other_backbones_sizes(list_net_sizes)
    reduced_df = complete_df[list_net_sizes_without_other_backbones]
    return reduced_df

def get_random_dataframes_from_plot_data(net_name, net_size, sparsifier_name, subgraphs_type, info_code):
    means, stds = get_data_plot_from_files(net_name, sparsifier_name, subgraphs_type, info_code)
    info = get_info_from_plot_name(net_name, sparsifier_name, subgraphs_type, info_code)
    list_net_sizes = info['list_net_sizes']
    list_net_sizes_labels = get_labels_from_ticks(list_net_sizes, net_size)
    fcs = info['fcs']

    complete_df = pd.DataFrame([list_net_sizes_labels, means, stds, fcs], columns=list_net_sizes, index=['label', 'r_inf_means', 'r_inf_stds', 'fcs'])

    umb_size_index, eb_size_index, mb_size_index, pb_size_index = list_net_sizes.index(info['um']), list_net_sizes.index(info['e']), list_net_sizes.index(info['m']), list_net_sizes.index(info['p'])
    
    umb_net_sizes = get_ticks_without_other_backbones_sizes(list_net_sizes[umb_size_index:])
    eb_net_sizes = get_ticks_without_other_backbones_sizes(list_net_sizes[eb_size_index:])
    mb_net_sizes = get_ticks_without_other_backbones_sizes(list_net_sizes[mb_size_index:])
    pb_net_sizes = get_ticks_without_other_backbones_sizes(list_net_sizes[pb_size_index:])

    rs_umb_reduced_df, rs_eb_reduced_df, rs_mb_reduced_df, rs_pb_reduced_df = complete_df[umb_net_sizes], complete_df[eb_net_sizes], complete_df[mb_net_sizes], complete_df[pb_net_sizes]

    return rs_umb_reduced_df, rs_eb_reduced_df, rs_mb_reduced_df, rs_pb_reduced_df

#%%
def sir_plot(net_name, plot_type, with_error, list_of_infos, beta, backbone=''):
    current_directory = os.getcwd()
    #Each element of list_of_infos is of the kind (x_values, y_values, errors, with_error, fcs, label, marker, linestyle, color)

    fig, ax = plt.subplots()
    ax.set_xlabel('$\chi$')
    #ax.axhline(y=1, color='black', linestyle='-')
    #ax.axhline(y=0, color='black', linestyle=':')

    ax.set_ylabel(r'$R^{\infty} (\%)$')
    
    if plot_type=='backbone_threshold_random':
        ax2 = ax.twinx()
        ax2.set_ylabel("Fraction of Connected Subgraphs")

    if plot_type=='backbones_distortion' or plot_type=='backbones_random':
        colors = ['#160b39', '#85216b', '#e65d2f', '#f6d746']
        list_of_infos=list(list_of_infos)
        for i in range(len(list_of_infos)):
            list_of_infos[i]=list(list_of_infos[i])
            list_of_infos[i][-1]=colors[i]
            list_of_infos[i]=tuple(list_of_infos[i])
        list_of_infos = tuple(list_of_infos)
    
    for info in list_of_infos:
        x_values, x_labels, y_values, errors, fcs, label, marker, linestyle, color = info
        if with_error:
            ax.errorbar(x_values, y_values, errors, capsize=7, marker = marker, linestyle=linestyle, markersize = 10, alpha=1, color=color, label=label)
        else:
            ax.plot(x_values, y_values, marker=marker, linestyle=linestyle, markersize = 10, alpha=1, color=color, label=label)
        
        if plot_type=='backbones_distortion' or plot_type=='backbones_random':
            ax.plot([x_values[0], x_values[0]], [0, y_values.tolist()[0]], color=color, linestyle='--', alpha=0.5, label=r'$\chi$='+str(x_values[0]))
        if label=='Threshold Proximity - Random Subgraphs':
            ax2.bar([x-40 for x in x_values], fcs, alpha=0.2, width=80, color=color)
        if label=='Random Subgraph - Random Subgraphs':
            ax2.bar([x+40 for x in x_values], fcs, alpha=0.2, width=80, color=color)

    ticks = max([info[0] for info in list_of_infos], key= lambda x:len(x))
    labels = max([info[1] for info in list_of_infos], key= lambda x:len(x))
    new_ticks = get_ticks_with_min_spacing(ticks, min_spacing=500)
    new_labels_indices = [ticks.tolist().index(x) for x in new_ticks]
    new_labels = [labels.tolist()[i] for i in new_labels_indices]
    ax.set_xticks(ticks=new_ticks, labels=new_labels)
    
    if plot_type=='backbones_distortion' or plot_type=='backbones_random':
        ax.set_ybound(lower=0)
        fig.set_size_inches(10, (6/8)*10)
        #fig.set_size_inches(30, (1/3)*30)
    if plot_type=='distortion_vs_random':
        backbone='_'+backbone
        fig.set_size_inches(10, (6/8)*10)
    if plot_type=='backbone_threshold_random':
        backbone='_'+backbone
        fig.set_size_inches(10, (6/8)*10)

    ax.legend()
    ax.set_title(r'$\beta$='+str(beta))
    
    if with_error:
        fig.savefig(current_directory+f'/{net_name}/sir/plots/{net_name}_beta{beta}_with_error_{plot_type}{backbone}.pdf', dpi=500, bbox_inches="tight")
    else:
        fig.savefig(current_directory+f'/{net_name}/sir/plots/{net_name}_beta{beta}_{plot_type}{backbone}.pdf', dpi=500, bbox_inches="tight")
    plt.show()
    #plt.close(fig)

#%%
if __name__ == '__main__':

    net_name = 'frenchHSNet'
    #net_name = 'exhibitNet'
    #net_name = 'workplaceNet'
    #net_name = 'medellinNet'
    #net_name = 'manizalesNet'

    beta=3

    net_size = nx.number_of_edges(nx.read_graphml(net_name+'/'+net_name+'.gml'))

    plots_specs = {'backbone_marker':'.',
                   'backbone_linestyle':'-',
                   'threshold_marker':'^',
                   'threshold_linestyle':'-.',
                   'threshold_color':'blue',
                   'random_marker':'v',
                   'random_linestyle':':',
                   'random_color':'green'}
    
    umb_d_complete_df, umb_d_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'ultrametric_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    umb_r_complete_df, umb_r_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'ultrametric_backbone', 'random', net_name_to_info_code[net_name][beta])

    eb_d_complete_df, eb_d_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'euclidean_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    eb_r_complete_df, eb_r_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'euclidean_backbone', 'random', net_name_to_info_code[net_name][beta])

    mb_d_complete_df, mb_d_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'metric_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    mb_r_complete_df, mb_r_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'metric_backbone', 'random', net_name_to_info_code[net_name][beta])

    pb_d_complete_df, pb_d_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'product_backbone', 'distortion', net_name_to_info_code[net_name][beta])
    pb_r_complete_df, pb_r_reduced_df = get_backbone_dataframes_from_plot_data(net_name, net_size, 'product_backbone', 'random', net_name_to_info_code[net_name][beta])

    th_umb_reduced_df = get_threshold_dataframe_from_plot_data(net_name, net_size, 'threshold_proximity', 'random', 'umb_'+net_name_to_info_code[net_name][beta])
    th_eb_reduced_df = get_threshold_dataframe_from_plot_data(net_name, net_size, 'threshold_proximity', 'random', 'eb_'+net_name_to_info_code[net_name][beta])
    th_mb_reduced_df = get_threshold_dataframe_from_plot_data(net_name, net_size, 'threshold_proximity', 'random', 'mb_'+net_name_to_info_code[net_name][beta])
    th_pb_reduced_df = get_threshold_dataframe_from_plot_data(net_name, net_size, 'threshold_proximity', 'random', 'pb_'+net_name_to_info_code[net_name][beta])

    rs_umb_reduced_df, rs_eb_reduced_df, rs_mb_reduced_df, rs_pb_reduced_df = get_random_dataframes_from_plot_data(net_name, net_size, 'random_subgraph', 'random', net_name_to_info_code[net_name][beta])
    
    #r_inf
    umb_d_t_half_plot_info = (umb_d_reduced_df.columns, umb_d_reduced_df.loc['label',:], umb_d_reduced_df.loc['r_inf_means',:], umb_d_reduced_df.loc['r_inf_stds',:], None, 'Ultra-Metric Backbone - Distortion Threshold', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'orange')
    umb_r_t_half_plot_info = (umb_r_reduced_df.columns, umb_r_reduced_df.loc['label',:], umb_r_reduced_df.loc['r_inf_means',:], umb_r_reduced_df.loc['r_inf_stds',:], None, 'Ultra-Metric Backbone - Random Subgraphs', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'red')
    eb_d_t_half_plot_info = (eb_d_reduced_df.columns, eb_d_reduced_df.loc['label',:], eb_d_reduced_df.loc['r_inf_means',:], eb_d_reduced_df.loc['r_inf_stds',:], None, 'Euclidean Backbone - Distortion Threshold', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'orange')
    eb_r_t_half_plot_info = (eb_r_reduced_df.columns, eb_r_reduced_df.loc['label',:], eb_r_reduced_df.loc['r_inf_means',:], eb_r_reduced_df.loc['r_inf_stds',:], None, 'Euclidean Backbone - Random Subgraphs', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'red')
    mb_d_t_half_plot_info = (mb_d_reduced_df.columns, mb_d_reduced_df.loc['label',:], mb_d_reduced_df.loc['r_inf_means',:], mb_d_reduced_df.loc['r_inf_stds',:], None, 'Metric Backbone - Distortion Threshold', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'orange')
    mb_r_t_half_plot_info = (mb_r_reduced_df.columns, mb_r_reduced_df.loc['label',:], mb_r_reduced_df.loc['r_inf_means',:], mb_r_reduced_df.loc['r_inf_stds',:], None, 'Metric Backbone - Random Subgraphs', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'red')
    pb_d_t_half_plot_info = (pb_d_reduced_df.columns, pb_d_reduced_df.loc['label',:], pb_d_reduced_df.loc['r_inf_means',:], pb_d_reduced_df.loc['r_inf_stds',:], None, 'Product Backbone - Distortion Threshold', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'orange')
    pb_r_t_half_plot_info = (pb_r_reduced_df.columns, pb_r_reduced_df.loc['label',:], pb_r_reduced_df.loc['r_inf_means',:], pb_r_reduced_df.loc['r_inf_stds',:], None, 'Product Backbone - Random Subgraphs', plots_specs['backbone_marker'], plots_specs['backbone_linestyle'], 'red')

    th_umb_t_half_plot_info = (th_umb_reduced_df.columns, th_umb_reduced_df.loc['label',:], th_umb_reduced_df.loc['r_inf_means',:], th_umb_reduced_df.loc['r_inf_stds',:], th_umb_reduced_df.loc['fcs',:], 'Threshold Proximity - Random Subgraphs', plots_specs['threshold_marker'], plots_specs['threshold_linestyle'], plots_specs['threshold_color'])
    th_eb_t_half_plot_info = (th_eb_reduced_df.columns, th_eb_reduced_df.loc['label',:], th_eb_reduced_df.loc['r_inf_means',:], th_eb_reduced_df.loc['r_inf_stds',:], th_eb_reduced_df.loc['fcs',:], 'Threshold Proximity - Random Subgraphs', plots_specs['threshold_marker'], plots_specs['threshold_linestyle'], plots_specs['threshold_color'])
    th_mb_t_half_plot_info = (th_mb_reduced_df.columns, th_mb_reduced_df.loc['label',:], th_mb_reduced_df.loc['r_inf_means',:], th_mb_reduced_df.loc['r_inf_stds',:], th_mb_reduced_df.loc['fcs',:], 'Threshold Proximity - Random Subgraphs', plots_specs['threshold_marker'], plots_specs['threshold_linestyle'], plots_specs['threshold_color'])
    th_pb_t_half_plot_info = (th_pb_reduced_df.columns, th_pb_reduced_df.loc['label',:], th_pb_reduced_df.loc['r_inf_means',:], th_pb_reduced_df.loc['r_inf_stds',:], th_pb_reduced_df.loc['fcs',:], 'Threshold Proximity - Random Subgraphs', plots_specs['threshold_marker'], plots_specs['threshold_linestyle'], plots_specs['threshold_color'])

    rs_umb_t_half_plot_info = (rs_umb_reduced_df.columns, rs_umb_reduced_df.loc['label',:], rs_umb_reduced_df.loc['r_inf_means',:], rs_umb_reduced_df.loc['r_inf_stds',:], rs_umb_reduced_df.loc['fcs',:], 'Random Subgraph - Random Subgraphs', plots_specs['random_marker'], plots_specs['random_linestyle'], plots_specs['random_color'])
    rs_eb_t_half_plot_info = (rs_eb_reduced_df.columns, rs_eb_reduced_df.loc['label',:], rs_eb_reduced_df.loc['r_inf_means',:], rs_eb_reduced_df.loc['r_inf_stds',:], rs_eb_reduced_df.loc['fcs',:], 'Random Subgraph - Random Subgraphs', plots_specs['random_marker'], plots_specs['random_linestyle'], plots_specs['random_color'])
    rs_mb_t_half_plot_info = (rs_mb_reduced_df.columns, rs_mb_reduced_df.loc['label',:], rs_mb_reduced_df.loc['r_inf_means',:], rs_mb_reduced_df.loc['r_inf_stds',:], rs_mb_reduced_df.loc['fcs',:], 'Random Subgraph - Random Subgraphs', plots_specs['random_marker'], plots_specs['random_linestyle'], plots_specs['random_color'])
    rs_pb_t_half_plot_info = (rs_pb_reduced_df.columns, rs_pb_reduced_df.loc['label',:], rs_pb_reduced_df.loc['r_inf_means',:], rs_pb_reduced_df.loc['r_inf_stds',:], rs_pb_reduced_df.loc['fcs',:], 'Random Subgraph - Random Subgraphs', plots_specs['random_marker'], plots_specs['random_linestyle'], plots_specs['random_color'])
    
        
    ##########################################################################################################################################################
    ########################################################## DISTORTION VS RANDOM SUBGRAPHS ################################################################
    ##########################################################################################################################################################
    
    #ULTRAMETRIC BACKBONE - WITH ERRORS
    sir_plot(net_name, 'distortion_vs_random', True, [umb_d_t_half_plot_info, umb_r_t_half_plot_info], beta, 'umb')
    #ULTRAMETRIC BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'distortion_vs_random', False, [umb_d_t_half_plot_info, umb_r_t_half_plot_info], beta, 'umb')
    
    #EUCLIDEAN BACKBONE - WITH ERRORS
    sir_plot(net_name, 'distortion_vs_random', True, [eb_d_t_half_plot_info, eb_r_t_half_plot_info], beta, 'eb')
    #EUCLIDEAN BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'distortion_vs_random', False, [eb_d_t_half_plot_info, eb_r_t_half_plot_info], beta, 'eb')
    
    #METRIC BACKBONE - WITH ERRORS
    sir_plot(net_name, 'distortion_vs_random', True, [mb_d_t_half_plot_info, mb_r_t_half_plot_info], beta, 'mb')
    #METRIC BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'distortion_vs_random', False, [mb_d_t_half_plot_info, mb_r_t_half_plot_info], beta, 'mb')
    
    #PRODUCT BACKBONE - WITH ERRORS
    sir_plot(net_name, 'distortion_vs_random', True, [pb_d_t_half_plot_info, pb_r_t_half_plot_info], beta, 'pb') 
    #PRODUCT BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'distortion_vs_random', False, [pb_d_t_half_plot_info, pb_r_t_half_plot_info], beta, 'pb')
    
   

    ##########################################################################################################################################################
    ################################################################# BACKBONE RANDOM THRESHOLD ##############################################################
    ##########################################################################################################################################################

    #ULTRAMETRIC BACKBONE - WITH ERRORS
    sir_plot(net_name, 'backbone_threshold_random', True, [umb_r_t_half_plot_info, th_umb_t_half_plot_info, rs_umb_t_half_plot_info], beta, 'umb_r')
    sir_plot(net_name, 'backbone_threshold_random', True, [umb_d_t_half_plot_info, th_umb_t_half_plot_info, rs_umb_t_half_plot_info], beta, 'umb_d')
    
    #ULTRAMETRIC BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'backbone_threshold_random', False, [umb_r_t_half_plot_info, th_umb_t_half_plot_info, rs_umb_t_half_plot_info], beta, 'umb_r')
    sir_plot(net_name, 'backbone_threshold_random', False, [umb_d_t_half_plot_info, th_umb_t_half_plot_info, rs_umb_t_half_plot_info], beta, 'umb_d')
    
    #EUCLIDEAN BACKBONE - WITH ERRORS
    sir_plot(net_name, 'backbone_threshold_random', True, [eb_r_t_half_plot_info, th_eb_t_half_plot_info, rs_eb_t_half_plot_info], beta, 'eb_r')
    sir_plot(net_name, 'backbone_threshold_random', True, [eb_d_t_half_plot_info, th_eb_t_half_plot_info, rs_eb_t_half_plot_info], beta, 'eb_d')
    #EUCLIDEAN BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'backbone_threshold_random', False, [eb_r_t_half_plot_info, th_eb_t_half_plot_info, rs_eb_t_half_plot_info], beta, 'eb_r')
    sir_plot(net_name, 'backbone_threshold_random', False, [eb_d_t_half_plot_info, th_eb_t_half_plot_info, rs_eb_t_half_plot_info], beta, 'eb_d')
    
    #METRIC BACKBONE - WITH ERRORS
    sir_plot(net_name, 'backbone_threshold_random', True, [mb_r_t_half_plot_info, th_mb_t_half_plot_info, rs_mb_t_half_plot_info], beta, 'mb_r')
    sir_plot(net_name, 'backbone_threshold_random', True, [mb_d_t_half_plot_info, th_mb_t_half_plot_info, rs_mb_t_half_plot_info], beta, 'mb_d')

    #METRIC BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'backbone_threshold_random', False, [mb_r_t_half_plot_info, th_mb_t_half_plot_info, rs_mb_t_half_plot_info], beta, 'mb_r')
    sir_plot(net_name, 'backbone_threshold_random', False, [mb_d_t_half_plot_info, th_mb_t_half_plot_info, rs_mb_t_half_plot_info], beta, 'mb_d')  

    #PRODUCT BACKBONE - WITH ERRORS
    sir_plot(net_name, 'backbone_threshold_random', True, [pb_r_t_half_plot_info, th_pb_t_half_plot_info, rs_pb_t_half_plot_info], beta, 'pb_r')
    sir_plot(net_name, 'backbone_threshold_random', True, [pb_d_t_half_plot_info, th_pb_t_half_plot_info, rs_pb_t_half_plot_info], beta, 'pb_d')
    #PRODUCT BACKBONE - WITHOUT ERRORS
    sir_plot(net_name, 'backbone_threshold_random', False, [pb_r_t_half_plot_info, th_pb_t_half_plot_info, rs_pb_t_half_plot_info], beta, 'pb_r')
    sir_plot(net_name, 'backbone_threshold_random', False, [pb_d_t_half_plot_info, th_pb_t_half_plot_info, rs_pb_t_half_plot_info], beta, 'pb_d')
    
    ##########################################################################################################################################################
    ######################################################################## BACKBONES #######################################################################
    ##########################################################################################################################################################

    #ALL BACKBONES RANDOM SUBGRAPHS - WITH ERRORS
    sir_plot(net_name, 'backbones_random', True, [umb_r_t_half_plot_info, eb_r_t_half_plot_info, mb_r_t_half_plot_info, pb_r_t_half_plot_info], beta)
    #ALL BACKBONES RANDOM SUBGRAPHS - WITHOUT ERRORS
    sir_plot(net_name, 'backbones_random', False, [umb_r_t_half_plot_info, eb_r_t_half_plot_info, mb_r_t_half_plot_info, pb_r_t_half_plot_info], beta)

    #ALL BACKBONES DISTORTION THRESHOLD - WITH ERRORS
    sir_plot(net_name, 'backbones_distortion', True, [umb_d_t_half_plot_info, eb_d_t_half_plot_info, mb_d_t_half_plot_info, pb_d_t_half_plot_info], beta)
    #ALL BACKBONES DISTORTION THRESHOLD - WITHOUT ERRORS
    sir_plot(net_name, 'backbones_distortion', False, [umb_d_t_half_plot_info, eb_d_t_half_plot_info, mb_d_t_half_plot_info, pb_d_t_half_plot_info], beta)

    ##########################################################################################################################################################