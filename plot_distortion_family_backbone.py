import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

code_to_name = {'D': 'Dombi','AA':'Aczél-Alsina', 'SS4': 'Schweizer&Sklar 4'}

for net_name in ['workplaceNet', 'exhibitNet', 'frenchHSNet']:
    for backbone_family in ['D', 'AA','SS4']:
        if backbone_family == 'SS4':
            net = nx.read_graphml(f'/Users/Bernardo/Desktop/thesis/{net_name}/m_{net_name}.gml')
        if backbone_family in ['D', 'AA']:
            net = nx.read_graphml(f'/Users/Bernardo/Desktop/thesis/{net_name}/um_{net_name}.gml')

        distortion_dic = nx.get_edge_attributes(net, 'distortion')
        
        df = pd.read_csv(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/phi_{backbone_family}_backbones_edgelists_1_05_025.csv')

        order = 'reverse'
        #order = 'normal'
        original_columns = list(df.columns)
        columns = list(df.columns)

        if order == 'reverse':
            columns.reverse()
            init_param = columns[-1]
        else:
            init_param = columns[0]

        init_param_value = df[init_param].name
        init_param_backbone_edges = [eval(edge.replace('inf','np.inf'))[:2] for edge in list(df[init_param]) if eval(edge.replace('inf','np.inf'))[2]==1.0]
        prevG = nx.Graph(init_param_backbone_edges)

        param_dic = {}
        for col in columns:
            current_param_value = df[col].name
            print(current_param_value)
            current_param_backbone_edges = [eval(edge.replace('inf','np.inf'))[:2] for edge in list(df[col]) if eval(edge.replace('inf','np.inf'))[2]==1.0]
            G = nx.Graph(current_param_backbone_edges)
            #print('difference_size:', nx.number_of_edges(nx.difference(G, prevG)))
            diff_edges = list(nx.difference(G, prevG).edges)
            for edge in diff_edges:
                if edge not in param_dic.keys():
                    param_dic[edge] = current_param_value
            prevG = G.copy()

        points = []
        for key in param_dic.keys():
            param = param_dic[key]
            try:
                distortion = distortion_dic[key]
            except:
                distortion = distortion_dic[(key[1],key[0])]
            points.append((distortion, param))

        points.sort(key=lambda x: x[0])

        columns = list(df.columns)
        param_to_int = dict(list(zip(columns, range(1,len(columns)+1))))
        int_to_param = dict(list(zip(range(1,len(columns)+1),columns)))
        new_points = [(point[0], param_to_int[point[1]]) for point in points]

        dists, params = zip(*new_points)

        ############################################################################################################
        
        fig, ax = plt.subplots(figsize=(8, (6/8)*8))

        ax.scatter(params, dists, marker='.', color='red')

        my_ticks = [tick for tick in list(ax.get_xticks()) if tick in int_to_param.keys()]
        print(my_ticks)
        my_labels = [int_to_param[int(tick)] for tick in my_ticks]
        print(my_labels)
        
        ax.set_xticks(ticks=my_ticks, labels=my_labels)
        
        if backbone_family == 'SS4':
            ax.set_ylabel('Metric Distortion'+r' $s^{m}$')
        if backbone_family in ['D', 'AA']:
            ax.set_ylabel('Ultra-Metric Distortion'+r' $s^{um}$')
        
        ax.set_xlabel(code_to_name[backbone_family]+r' $\lambda$')

        fig.tight_layout()
        fig.savefig(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/{net_name}_lambda_vs_distortion_{backbone_family}_new.pdf', dpi=300)
        plt.show()

        ############################################################################################################

        fig, ax = plt.subplots(figsize=(8, (6/8)*8))

        ax.scatter(dists, params, marker='.', color='red')

        my_ticks = [tick for tick in list(ax.get_yticks()) if tick in int_to_param.keys()]
        print(my_ticks)
        my_labels = [int_to_param[int(tick)] for tick in my_ticks]
        print(my_labels)
        
        ax.set_yticks(ticks=my_ticks, labels=my_labels)
        
        if backbone_family == 'SS4':
            ax.set_xlabel('Metric Distortion'+r' $s^{m}$')
        if backbone_family in ['D', 'AA']:
            ax.set_xlabel('Ultra-Metric Distortion'+r' $s^{um}$')
        
        ax.set_ylabel(code_to_name[backbone_family]+r' $\lambda$')

        fig.tight_layout()
        fig.savefig(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/{net_name}_distortion_vs_lambda_{backbone_family}_new.pdf', dpi=300)
        plt.show()
    
        ############################################################################################################
        '''
        edge_distortion_list = [(edge, distortion_dic[edge]) for edge in list(distortion_dic.keys()) if distortion_dic[edge]>1]
        edge_distortion_list.sort(key=lambda x: x[1])
        distortion_order_dic = dict([(frozenset(edge_distortion_list[i][0]), i+1) for i in range(len(edge_distortion_list))])

        new_param_dic = dict([(frozenset(edge), param_dic[edge]) for edge in param_dic.keys()])

        distortion_order_lambda_points = [(distortion_order_dic[edge], param_to_int[new_param_dic[edge]]) for edge in distortion_order_dic.keys()]

        dist_order, lambda_int = zip(*distortion_order_lambda_points)

        fig, ax = plt.subplots(figsize=(8, (6/8)*8))

        ax.scatter(dist_order, lambda_int, marker='.', color='red', s=1)

        my_ticks = [tick for tick in list(ax.get_yticks()) if tick in int_to_param.keys()]
        print(my_ticks)
        my_labels = [int_to_param[int(tick)] for tick in my_ticks]
        print(my_labels)
        ax.set_yticks(ticks=my_ticks, labels=my_labels)
        
        if backbone_family == 'SS4':
            ax.set_xlabel('Metric Distortion Edge Order')
        if backbone_family in ['D', 'AA']:
            ax.set_xlabel('Ultra-Metric Distortion Edge Order')
        
        ax.set_ylabel(code_to_name[backbone_family]+r' $\lambda$')

        fig.tight_layout()
        fig.savefig(f'/Users/Bernardo/Desktop/thesis/{net_name}/backbones_sizes/{net_name}_distortion_order_vs_lambda_{backbone_family}_new.pdf', dpi=300)
        plt.show()
        '''





