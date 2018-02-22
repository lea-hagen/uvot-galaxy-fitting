import numpy as np
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
import pickle
import pdb

import model_parameters
import best_val_chi2

def triangle(file_label, verbose=False):
    """
    Make a nice colorful triangle plot showing the 2D probability distributions


    Parameters
    ----------
    file_label : string
        the label associated with the region/galaxy
        
    verbose : boolean
        set to True to print out various diagnostic statements

    Returns
    -------
    nothing

    """


    # load the chi2 grid
    pickle_file = open('./pickles/chi2_'+file_label+'.pickle','rb')
    results = pickle.load(pickle_file)
    pickle_file.close()

    # load the best fit values
    best_fit = best_val_chi2.best_val(file_label, verbose=False)

    # indices to plot (accounting for parameters held constant)
    plot_index = []
    for i,axis in enumerate(results['axis_order']):
        if len(results['grid_axes'][axis]) > 1:
            plot_index.append(i)
    # number of dimensions to fit
    n_dimen = len(plot_index)
    # total number of parameters
    n_param = len(results['axis_order'])

    # calculate ln(P) from chi^2
    ln_p = -0.5 * results['new_chi2_grid']
        
    
    fig = plt.figure(figsize=(10,10))
    #fig.subplots_adjust(hspace=0.05)
    #fig, ax = plt.subplots(n_dimen-1, n_dimen-1, sharex=True, figsize=(10,10))

    for i,pi in enumerate(plot_index):


        for j,pj in enumerate(plot_index[i:], i):

 
            #ax[j,i].set_xlim([-9,9])
            #ax[j,i].set_ylim([-9,9])

            axis_label_i = results['axis_order'][pi]
            best_val_i = best_fit[axis_label_i[:-5]]
            axis_label_j = results['axis_order'][pj]
            best_val_j = best_fit[axis_label_j[:-5]]

            if i != j:

                plt.subplot(n_dimen, n_dimen, i + j*(n_dimen) + 1)
                ax = plt.gca()


                if verbose == True:
                    print('')
                    print('     plotting x='+axis_label_i + ', y='+axis_label_j)
                    print('     i='+str(i) + ', j='+str(j))

                # smash the ln_p array along the other dimensions
                axis_smash = np.delete(np.arange(n_param), [pi,pj]) 
                plot_ln_p = np.log( np.sum(np.exp(ln_p), axis=tuple(axis_smash)) )
                # set -inf to the minimum non-inf value
                plot_ln_p[plot_ln_p == -np.inf] = np.min(plot_ln_p[plot_ln_p != -np.inf])
                # apparently also have to rotate the array
                plot_ln_p = np.rot90(plot_ln_p, 1)
                if verbose == True:
                    print('        ln_p dimensions: ', plot_ln_p.shape)
                    print('        ln_p range: ', np.min(plot_ln_p), np.max(plot_ln_p))

                # debugging: do a manual ln_p
                #plot_ln_p = np.sum(np.indices(plot_ln_p.shape), axis=0)
                
                # figure out the axis ranges
                x_range = [np.min(results['grid_axes'][axis_label_i]),
                               np.max(results['grid_axes'][axis_label_i]) ]
                y_range = [np.min(results['grid_axes'][axis_label_j]),
                               np.max(results['grid_axes'][axis_label_j]) ]
                extent = x_range[0], x_range[1], y_range[0], y_range[1]
                if verbose == True:
                    print('        extent=', extent) 
                
                # plotting
                #ax[j,i].imshow(plot_ln_p, cmap='hot', interpolation='nearest',
                #                   aspect=(x_range[1]-x_range[0])/(y_range[1]-y_range[0]), extent=extent)
                
                plt.imshow(plot_ln_p, cmap='hot', interpolation='nearest', #vmin=-600,
                                   aspect=(x_range[1]-x_range[0])/(y_range[1]-y_range[0]), extent=extent)
                # and best fit
                plt.plot(best_val_i, best_val_j, marker='x', c='limegreen',
                             markersize=12, markeredgewidth=5)
                
                
                # debugging labels
                #plt.text(0.5, 0.1, axis_label_i[:-5], ha='center',va='center', transform=ax.transAxes)
                #plt.text(0.1, 0.5, axis_label_j[:-5], ha='center',va='center', transform=ax.transAxes)
                
                # axis labels
                if i == 0:
                    ax.set_ylabel(axis_label_j[:-5])
                    if verbose == True:
                        print('        adding y-axis label ', axis_label_j[:-5])
                else:
                    ax.set_yticklabels([])
                if j == n_dimen-1:
                    ax.set_xlabel(axis_label_i[:-5])
                    if verbose == True:
                        print('        adding x-axis label ', axis_label_i[:-5])
                else:
                    ax.set_xticklabels([])

                #if (i == 0) and (j == 3):
                #    pdb.set_trace()

            if i == j:

                if verbose == True:
                    print('')
                    print('     plotting x='+axis_label_i)
                    print('     i='+str(i) + ', j='+str(j))
                
                plt.subplot(n_dimen, n_dimen, i + j*(n_dimen) + 1)
                ax = plt.gca()

                # smash the ln_p array along the other dimensions
                axis_smash = np.delete(np.arange(n_param), [pi]) 
                plot_ln_p = np.log( np.sum(np.exp(ln_p), axis=tuple(axis_smash)) )

                # do some plotting
                ax.plot(results['grid_axes'][axis_label_i], plot_ln_p,
                             marker='.', markersize=10)
                plt.xlim(np.min(results['grid_axes'][axis_label_i]), np.max(results['grid_axes'][axis_label_i]))
                ax.set_yticklabels([])
                
                
        # clear non-plotted areas
        #for j in range(0,i+1):
        #    ax.get_xaxis().set_visible(False)
        #    ax.get_yaxis().set_visible(False)

        
    fig.savefig('./plots/triangle_'+file_label+'.pdf')
    plt.close(fig)

    #pdb.set_trace()
