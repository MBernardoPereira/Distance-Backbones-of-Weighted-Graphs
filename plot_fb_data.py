import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#colors = list(mcolors.TABLEAU_COLORS.values())

def integer_values(str_k_values):
    #min_index = 0
    one_index = str_k_values.index('1') #str_k_values.index('1.0')
    max_index = len(str_k_values)-1
    res = [i for i in range(-one_index, max_index-one_index+1)]
    return res

def create_backbones_sizes_plot(net_name, backbones_info, l1, l2, g_size, pb_size, mb_size, umb_size):
    fig, ax1 = plt.subplots()
    
    phi_family_name = {'phi_D': 'Dombi',
                       'phi_AA': 'Aczel-Alsina',
                       'phi_T': 'Trigonometric',
                       'phi_F': 'Frank',
                       'phi_H': 'Hamacher',
                       'phi_SS4': 'Schweiser&Sklar 4'}
    
    phi_color = {'phi_D':  '#1f77b4',
                 'phi_AA': '#ff7f0e',
                 'phi_T':  '#2ca02c',
                 'phi_F':  '#d62728',
                 'phi_H':  '#9467bd',
                 'phi_SS4': '#8c564b',
                 'phi_T2': '#e377c2'}
    
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ks = backbones_info[0][1]
    ks.sort(key = lambda x: eval(x))
    i1 = ks.index(l1)
    i2 = ks.index(l2)
    
    x_values = integer_values(ks)
    str_to_int = dict([(k_str, x_value) for (k_str, x_value) in zip(ks, x_values)])
    
    for backbone_info in backbones_info:
        backbone_name, ks_list, sizes_list = backbone_info
        backbone_x_values = [str_to_int[k] for k in ks_list]
        ax1.plot(backbone_x_values[i1:i2+1], sizes_list[i1:i2+1], marker = '.', linestyle='-', linewidth=1, color=phi_color[backbone_name], label=phi_family_name[backbone_name])
    
    ax1.legend(bbox_to_anchor=(0.6,0.7), loc="upper left", fontsize=12)
    ax1.set_xlabel(r'$\lambda$', fontsize=15)
    
    sparse_x_labels = [k for k in ks[i1:i2+1] if (eval(k.split('/')[-1]))%10==0 or k=='1']
    sparse_x_values = [str_to_int[k] for k in sparse_x_labels]
    ax1.set_xticks(sparse_x_values, sparse_x_labels, rotation=-45)
    ax1.tick_params(axis='x', labelsize=15)
   
    sparse_y_values = [umb_size, pb_size, mb_size, g_size]
    sparse_y_labels = [str(round(s*100/g_size,1)) for s in sparse_y_values]
    ax1.set_yticks(sparse_y_values, sparse_y_labels)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_ylabel('Percentage of Edges', fontsize=18)

    ax2 = ax1.twinx()
    ax2.axhline(y=g_size, color='black', linestyle='-', label='Network Size')
    ax2.axhline(y=pb_size, color='black', linestyle='-.', label='Product Backbone Size')
    ax2.axhline(y=mb_size, color='black', linestyle='--', label='Metric Backbone Size')
    ax2.axhline(y=umb_size, color='black', linestyle=':', label='Ultra-Metric Backbone Size')
    ax2.axvline(x=0, color='grey', linestyle='-', label=r'$\lambda=1$')
    ax2.legend(bbox_to_anchor=(0.6,0.25), loc="lower left", fontsize=12)

    sparse_y_labels2 = [str(s) for s in sparse_y_values]
    ax2.set_yticks(sparse_y_values, sparse_y_labels2)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylabel('Number of Edges', fontsize=18)
        
    #plt.title('Backbone Sizes of '+net_name)
    fig.set_size_inches(10, (6/8)*10)
    plt.tight_layout()
    plt.show()
    #Define the output directory and file name
    phis_str = '_'.join([el[0] for el in backbones_info])
    net_directory = os.getcwd() + f'/{net_name}'
    output_directory = net_directory+f'/backbones_sizes/NOW_{net_name}_backbones_sizes_{phis_str}_{l2}.pdf'
    fig.savefig(output_directory, dpi=300)


if __name__ == '__main__':

    important_sizes_dic = {'frenchHSNet':{'g_size': 5818, 'pb_size': 4369, 'mb_size': 603,'umb_size': 326},
                           'exhibitNet':{'g_size': 2765, 'pb_size': 2661, 'mb_size': 1088,'umb_size': 413},
                           'workplaceNet':{'g_size': 4274, 'pb_size': 4164, 'mb_size': 745,'umb_size': 216},
                           'hospitalNet':{'g_size': 1139, 'pb_size': 1024, 'mb_size': 217, 'umb_size': 74}, 
                           'conferenceNet':{'g_size': 2196, 'pb_size': 2027,'mb_size': 331, 'umb_size': 112},
                           'primaryschoolNet':{'g_size': 8317, 'pb_size': 7630, 'mb_size': 790, 'umb_size': 241},
                           'manizalesNet':{'g_size': 2518, 'pb_size': 2438, 'mb_size': 671, 'umb_size': 193},
                           'medellinNet':{'g_size': 33884, 'pb_size': 33730, 'mb_size': 8360, 'umb_size': 1844},
                           'undirUSairportsNet':{'g_size': 0, 'pb_size': 326, 'mb_size': 326, 'umb_size': 0}}

    #Pick a network from ['frenchHSNet', 'hospitalNet', 'conferenceNet', 'primaryschoolNet', 'undirUSairportsNet']
    for net_name in ['frenchHSNet', 'exhibitNet', 'workplaceNet']:
        net_directory = os.getcwd() + f'/{net_name}'

        #Choose phis to plot
        phis_to_plot = ['phi_D', 'phi_AA', 'phi_H', 'phi_F', 'phi_SS4'] #'phi_F'
        
        #Load data for every phis
        backbones_info = []
        for phi in phis_to_plot:
            print(phi)
            df = pd.read_csv(net_directory+f'/backbones_sizes/{phi}_backbones_sizes_1_05.csv')
            ks_list = df.columns.tolist()
            sizes_list = df.loc[0, :].values.flatten().tolist()
            backbones_info.append( (phi, ks_list, sizes_list) )
        
        #Load backbones sizes
        g_size = important_sizes_dic[net_name]['g_size']
        pb_size = important_sizes_dic[net_name]['pb_size'] 
        mb_size = important_sizes_dic[net_name]['mb_size']
        umb_size = important_sizes_dic[net_name]['umb_size']
        
        #Create plot
        l1 = '1/100'
        l2 = '100'
        create_backbones_sizes_plot(net_name, backbones_info, l1, l2, g_size, pb_size, mb_size, umb_size)

        #Create plot
        l1 = '1/20'
        l2 = '20'
        create_backbones_sizes_plot(net_name, backbones_info, l1, l2, g_size, pb_size, mb_size, umb_size)


#%%