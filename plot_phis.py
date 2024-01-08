import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
#from matplotlib.colors import Normalize
import os

def get_phi(phi_name, k):
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
    return phi_functions.get(phi_name)


phi_letter = {#Generators for Dombi T-Norms
                'phi_D': 'D',
                #Generators for Aczel-Alsina T-Norms
                'phi_AA': 'AA',
                #Generators for Trigonometric T-Norms
                'phi_T': 'T',
                #Generators for Frank T-Norms (The else case is because lim k->1 (k**x - 1)/(k-1) = x)
                'phi_F': 'F',
                #Generators for Hamacher T-Norms
                'phi_H': 'H',
                #Generators for Schweiser&Sklar4 T-Norms
                'phi_SS4': 'SS4',
                #Generators for Trigonometric 2 T-Norms
                'phi_T2': 'T2'}

phi_color = {'phi_D': '#1f77b4',
            'phi_AA': '#ff7f0e',
            'phi_T': '#2ca02c',
            'phi_F': '#d62728',
            'phi_H': '#9467bd',
            'phi_SS4': '#8c564b',
            'phi_T2': '#e377c2'}

def plot_phi_family(phi_family, y_upper_lim, directory, powers):
    x = np.linspace(0.00001, 1, 1000)
    ys = {k: get_phi(phi_family, k)(x) for k in powers}

    fig, ax = plt.subplots()

    cm = plt.get_cmap('viridis')

    normalized_data = np.linspace(0, 1, len(powers))
    val_to_norm = {key: value for key, value in zip(powers, normalized_data)}
    norm_to_color = {val_to_norm[k]: cm(val_to_norm[k]) for k in powers}

    for k in powers:
        color = cm(val_to_norm[k])
        if k==1:
            ax.plot(x, ys[k], color='black', label=r'$\lambda=1$')
        else:
            ax.plot(x, ys[k], color=color)

    ax.plot(x, ys[k], color=color)

    ax.set_title(r'$\varphi^{%s}_{\lambda}$' % (phi_letter[phi_family]), size=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_upper_lim)

    # Create a ScalarMappable to link the colormap to the data
    #divnorm = mpl.colors.TwoSlopeNorm(vmin=-1/100, vcenter=1, vmax=100)
    sm = ScalarMappable(cmap=cm)#, norm=divnorm)
    #sm.set_array([])  # You can set an empty array or None

    # Add a colorbar
    cb = plt.colorbar(sm, ax=ax)

    cb.set_label(r'$\lambda$', rotation=0, labelpad=0.1)  # Rotate label by 90 degrees and set labelpad

    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['1/100','1','100'])

    ax.legend()   

    fig.set_figheight(5)
    fig.set_figwidth(4)

    plt.savefig(directory + f'/plots_of_{phi_family}.pdf', dpi=500, bbox_inches="tight")
    plt.show()

def plot_important_phis(phis, y_upper_lim, directory):
    x = np.linspace(0.00001, 1, 1000)

    #ys_phi1 = get_phi('phi1', 1)(x)
    #ys_phi2 = get_phi('phi2', 1)(x)
    #ys_phi3 = get_phi('phi3', 1)(x)

    phi_parameters = {'phi_D': {'ys': get_phi('phi_D',1)(x), 'color': '#1f77b4', 'label' : r'$\varphi^{D}_1 = \varphi^{SS4}_1 = \frac{1}{x} - 1$'},
                      'phi_AA': {'ys': get_phi('phi_AA',1)(x), 'color': '#ff7f0e', 'label' : r'$\varphi^{AA}_1 = \varphi^{F}_1 = \varphi^{H}_1 = -\log(x)$'},
                      'phi_T': {'ys': get_phi('phi_T',1)(x), 'color': '#2ca02c', 'label' : r'$\varphi^{T}_1 = \varphi^{T2}_1 = -\tan(\frac{\pi}{2} (x-1))$'}}

    fig, ax = plt.subplots()

    for phi in phis:
        ax.plot(x, phi_parameters[phi]['ys'], color=phi_parameters[phi]['color'], label=phi_parameters[phi]['label'])

    #ax.set_title(r'$\varphi^k_{%s}$' % (phi_letter[phi_family]), size=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_upper_lim)

    ax.legend()        
    phis_list = '_'.join(phis)  

    fig.set_figheight(5)
    fig.set_figwidth(4)

    plt.savefig(directory + f'/plot_of_{phis_list}.pdf', dpi=500, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    powers = [1/n for n in np.arange(100,1,-1)]+[n for n in np.arange(1,101,1)]
    y_upper_lim = 10
    directory = os.getcwd()

    for phi_family in ['phi_D', 'phi_AA', 'phi_T', 'phi_F', 'phi_H', 'phi_SS4', 'phi_T2']:
        plot_phi_family(phi_family, y_upper_lim, directory, powers)
    
    phis=['phi_D', 'phi_AA']
    plot_important_phis(phis, y_upper_lim, directory)

    phis=['phi_D', 'phi_AA', 'phi_T']
    plot_important_phis(phis, y_upper_lim, directory)