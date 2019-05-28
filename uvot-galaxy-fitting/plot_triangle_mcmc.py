import numpy as np
import pickle
import matplotlib.pyplot as plt
import corner


def plot_triangle(file_label, chain, best_fit):
    """
    Create a simple triangle plot for the results

    Parameters
    ----------
    file_label : string
        the label associated with the region/galaxy
    
    chain : dict
        the trimmed and flattened chains (length N for fitted, length 1 for
        held constant), output from make_chains_mcmc.py

    best_fit : dict
        the dictionary with the best-fit values (and errors) output from best_val_mcmc.py


    Returns
    -------
    nothing

    """

    # figure out which chains to plot
    param_list = ['av','rv','bump','log_age','log_mass','tau']
    if 'log_mass_ratio' in chain.keys():
        param_list = ['av','rv','bump','log_age','log_mass','log_mass_ratio','tau']
        
    plot_param = [p for p in param_list if len(chain[p]) > 1]

    # need to put the chains into a giant array
    flatchain = np.zeros(( len(plot_param), len(chain[plot_param[0]]) ))
    for p, param in enumerate(plot_param):
        flatchain[p] = chain[param]
    flatchain = np.transpose(flatchain)

    # set "truths" to be the "best" value
    #truths = [best_fit[p+'_best'] for p in plot_param]
    
    # choose the sigmas for contours
    #sigmas = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    #c_levels = {'levels':1.0 - np.exp( -0.5 * sigmas **2 ) }
    
    fig = corner.corner(flatchain, labels=plot_param,
                            quantiles=[0.16,0.5,0.84], verbose=False)
                            #truths=truths)#, hist2d_kwargs=c_levels)
    plt.figure(figsize=(8,8))
    fig.savefig('./plots/triangle_'+file_label+'.pdf')
    plt.close(fig)


    
