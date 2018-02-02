from __future__ import print_function
import numpy as np
import scipy
import scipy.io
import scipy.interpolate
import emcee
import corner as triangle
#import cPickle as pickle  # (only 2.x)
import pickle
import matplotlib
import matplotlib.pyplot as plt
import datetime
import is_outlier#; reload(is_outlier)
import os.path
import pdb

import model_parameters

# ============================================
# HOW TO USE THIS FILE:
# import run_emcee; reload(run_emcee)
# run_emcee.run_emcee()
# ============================================



def run_chi2():
    """
    Run emcee on the grid models.
    """

    # choose whether to run emcee
    run_emcee = True
    #run_emcee = False

    # choose whether to make plots
    plot_triangle = True
    #plot_triangle = False
    plot_spectrum = True
    #plot_spectrum = False

    # choose whether to redo emcee for regions that have already been done
    re_run = True
    #re_run = False

    print('START TIME:')
    print(datetime.datetime.now())


    # -------------------------
    # set-up
    # -------------------------

    print('setting up the data...')

    # read in the pickle file with the model grid
    pickle_file = open('model_grid.pickle','rb')
    model_info = pickle.load(pickle_file)
    pickle_file.close()
    # change tau list to numpy array
    model_info['tau_list'] = np.array(model_info['tau_list']).astype(np.float)


    # read in the pickle file with magnitudes
    pickle_file = open(model_parameters.phot_file,'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    # put desired filters into an array
    n_region = len(data['label'])
    n_filter = len(model_info['filter_list'])
    data['mag_list'] = np.zeros((n_filter,n_region))
    data['mag_list_err'] = np.zeros((n_filter,n_region))
    for f,filt in enumerate(model_info['filter_list']):
        data['mag_list'][f,:] = data[filt]
        data['mag_list_err'][f,:] = data[filt+'_err']

    
    ## read in the IDL sav file with [adjusted] Draine+14 AVs
    #temp = scipy.io.readsav('../modeling_pix_av/d14_av_list.sav')
    #d14_av = temp['d14_av_list']
    ## divide the AVs in half
    #d14_av /= 2.0
    
    
    # also turn it into f_nu
    data['flux_list'] = 10**( (data['mag_list'] + 48.6)/(-2.5) )
    data['flux_list_err'] = data['flux_list'] / 2.5 * np.log(10) * data['mag_list_err']
    
    # the associated wavelengths (angstroms)
    lambda_list = model_info['lambda_list']

    #print(data['mag_list'])
    #print(data['mag_list_err'])


    
    # make a flux version of the model grid (for interpolating)
    model_info['model_fnu'] = 10**( (model_info['model_mags'] + 48.6)/(-2.5) )

    # generate the functions that will do grid interpolation for magnitudes
    grid_func = []
    for i in range(n_filter):
        grid_func.append( scipy.interpolate.RegularGridInterpolator(( model_info['tau_list'], 
                                                                      model_info['av_list'], 
                                                                      model_info['age_list'], 
                                                                      model_info['bump_list'], 
								                                      model_info['rv_list']), 
                                                                      model_info['model_fnu'][:,:,:,:,:,i], 'linear', 
                                                                      bounds_error=True, fill_value=None) )

    # also make functions for stellar mass
    mstar_grid_func = scipy.interpolate.RegularGridInterpolator( (model_info['tau_list'], model_info['age_list']), 
                                                                      model_info['model_stellar_mass'], 'linear', 
                                                                      bounds_error=True, fill_value=None)

    
    #print(type(grid_func))

    #pdb.set_trace()

    # remove the giant grids from the model_info dictionary
    del model_info['model_mags']
    #del model_info['model_fnu']
    del model_info['model_stellar_mass']
    del model_info['readme']

    

    # -------------------------
    # run emcee
    # -------------------------

    # number of variables we're fitting (grid + mass)
    n_dimen = len(model_info['model_fnu'].shape)

    # divide the magnitudes into groups so they can be run separately
    n_groups = 7
    mod_val = data['mag_list'].shape[1] / n_groups + 1   # +1 to account for remainder


    group_num = 6
    first_index = int( group_num*mod_val )
    last_index = int( (group_num+1)*mod_val-1 )
    #print(group_num*mod_val, (group_num+1)*mod_val-1)

    # label for files
    file_label = data['label']


    for choose_region in range(first_index,last_index+1):
    #for choose_region in range(128,133):
    #for choose_region in [1]:

        print('')
        print('****************************')
        print('STARTING REGION ', choose_region)
        print('(group ', group_num, ' does ', first_index, '-', last_index, ')')
        print('****************************')
        print('')
        print('START TIME:')
        print(datetime.datetime.now())
        print('')

        for i in range(n_filter):
            print(model_info['filter_list'][i], '{:.3f}'.format(data['mag_list'][i,choose_region]), '+/-',
                      '{:.3f}'.format(data['mag_list_err'][i,choose_region]))

        if (run_emcee == True) and \
            (re_run == True or \
            (re_run == False and os.path.isfile('./pickles/chi2_'+file_label[choose_region]+'.pickle') == False)):
            
            print('running chi2 fitter...')

    
            sub_data = {'mag_list':np.reshape(data['mag_list'][:,choose_region], -1), \
                        'mag_list_err':np.reshape(data['mag_list_err'][:,choose_region], -1), \
                        'flux_list':np.reshape(data['flux_list'][:,choose_region], -1), \
                        'flux_list_err':np.reshape(data['flux_list_err'][:,choose_region], -1) }

            results = chi2_grid(grid_func, model_info, sub_data)
            # results is dictionary with keys new_chi2_grid, new_mass_grid, grid_axes

            #pdb.set_trace()
            
            # save the chains from sampler object
            print('saving chains...')
            pickle_file = open('./pickles/chi2_'+file_label[choose_region]+'.pickle','wb')
            pickle.dump( results, pickle_file)
            pickle_file.close()
            del results


    
        # load the chains back in
        print('loading chi2 results...')
        
        pickle_file = open('./pickles/chi2_'+file_label[choose_region]+'.pickle','rb')
        results = pickle.load(pickle_file)
        pickle_file.close()
    
    
        print('END TIME:')
        print(datetime.datetime.now())
        print('')
    
        # -------------------------
        # extract best fit
        # -------------------------
    
        print('extracting best fits...')
    
        matplotlib.interactive(False)
    
    
        # test - remove outliers using log mass
        # (all of the outliers are outlying together)
        #bad = np.array( is_outlier.is_outlier(flatchain[:,3]) )
        #flatchain = flatchain[~bad,:]
         
        # find the best values

        
        best_val = np.empty(n_dimen)
        
        # a file to save the results
        results_file = open('./results/results_'+file_label[choose_region]+'.list','w')

        # get the indices for the best chi2
        best_chi2_index = np.where(results['new_chi2_grid'] == np.min(results['new_chi2_grid']))
        
        for i in range(n_dimen):

    
            # calculating
            if i < n_dimen-1:
                current_axis = results['axis_order'][i]
                best_val[i] = results['grid_axes'][current_axis][best_chi2_index[i][0]]
            if i == n_dimen-1:
                current_axis = 'log_mass'
                best_val[i] = results['new_mass_grid'][best_chi2_index][0]
            print('current_axis: ', current_axis)
            print('   Best value: ', '{:.3f}'.format(best_val[i]))
            results_file.write('   ' + '{:.3f}'.format(best_val[i]))

        best_tau, best_av, best_log_age, best_bump, best_rv, best_log_mass = best_val

        # calculate a stellar mass too
        print('calculating stellar mass...')
        best_mstar = np.log10( 10**best_val[-1] *
                                   mstar_grid_func(np.array([ best_tau, best_log_age ])) )
        

        print('Best mstar: ', '{:.3f}'.format(best_mstar[0]))
        results_file.write('   ' + '{:.3f}'.format(best_mstar[0]) )

                  
        # tau, av, log age, bump, rv, log mass
        print('   best tau, av, log_age, bump, rv, log_mass, log_stellar_mass:')
        print('   ', best_val, '  ', best_mstar)
        print('')

        # close the file
        results_file.write('\n')
        results_file.close()

        #pdb.set_trace()
    
        # -------------------------
        # analyze the results
        # -------------------------

        print('plotting...')

        # get it into probability space
        ln_p = -0.5 * results['new_chi2_grid']
        
    
        if plot_triangle == True:
            
    
            # make a triangle plot
            print('   triangle...')

            fig = plt.figure(figsize=(10,10))
            #fig.subplots_adjust(hspace=0.05)
            #fig, ax = plt.subplots(n_dimen-1, n_dimen-1, sharex=True, figsize=(10,10))

            for i in range(n_dimen-1):


                for j in range(i,n_dimen-1):

 
                    #ax[j,i].set_xlim([-9,9])
                    #ax[j,i].set_ylim([-9,9])

                    axis_label_i = results['axis_order'][i]
                    best_val_i = results['grid_axes'][axis_label_i][best_chi2_index[i]]
                    axis_label_j = results['axis_order'][j]
                    best_val_j = results['grid_axes'][axis_label_j][best_chi2_index[j]]

                    if i != j:

                        plt.subplot(n_dimen-1, n_dimen-1, i + j*(n_dimen-1) + 1)
                        ax = plt.gca()


                        print('')
                        print('     plotting x='+axis_label_i + ', y='+axis_label_j)
                        print('     i='+str(i) + ', j='+str(j))

                        # smash the ln_p array along the other dimensions
                        axis_smash = np.delete(np.arange(n_dimen-1), [i,j]) 
                        plot_ln_p = np.log( np.sum(np.exp(ln_p), axis=tuple(axis_smash)) )
                        # set -inf to the minimum non-inf value
                        plot_ln_p[plot_ln_p == -np.inf] = np.min(plot_ln_p[plot_ln_p != -np.inf])
                        # apparently also have to rotate the array
                        plot_ln_p = np.rot90(plot_ln_p, 1)
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
                        print('        extent=', extent)
                        
                        # plotting
                        #ax[j,i].imshow(plot_ln_p, cmap='hot', interpolation='nearest',
                        #                   aspect=(x_range[1]-x_range[0])/(y_range[1]-y_range[0]), extent=extent)
                        
                        plt.imshow(plot_ln_p, cmap='hot', interpolation='nearest', #vmin=-600,
                                           aspect=(x_range[1]-x_range[0])/(y_range[1]-y_range[0]), extent=extent)
                        # and best fit
                        plt.plot(best_val[i], best_val[j], marker='x', c='limegreen',
                                     markersize=12, markeredgewidth=5)
                        
                        
                        # debugging labels
                        #plt.text(0.5, 0.1, axis_label_i[:-5], ha='center',va='center', transform=ax.transAxes)
                        #plt.text(0.1, 0.5, axis_label_j[:-5], ha='center',va='center', transform=ax.transAxes)
                        
                        # axis labels
                        if i == 0:
                            ax.set_ylabel(axis_label_j[:-5])
                            print('        adding y-axis label ', axis_label_j[:-5])
                        else:
                            ax.set_yticklabels([])
                        if j == n_dimen-2:
                            ax.set_xlabel(axis_label_i[:-5])
                            print('        adding x-axis label ', axis_label_i[:-5])
                        else:
                            ax.set_xticklabels([])

                        #if (i == 0) and (j == 3):
                        #    pdb.set_trace()

                    if i == j:

                        print('')
                        print('     plotting x='+axis_label_i)
                        print('     i='+str(i) + ', j='+str(j))
                        
                        plt.subplot(n_dimen-1, n_dimen-1, i + j*(n_dimen-1) + 1)
                        ax = plt.gca()

                        # smash the ln_p array along the other dimensions
                        axis_smash = np.delete(np.arange(n_dimen-1), [i]) 
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

        
            fig.savefig('./plots/triangle_'+file_label[choose_region]+'.pdf')
            plt.close(fig)

            #pdb.set_trace()

    
        if plot_spectrum == True:
   
            # plot photometry
            print('   photometry...')
            plt.figure(figsize=(7,5))
            plt.errorbar( np.log10(lambda_list), data['mag_list'][:,choose_region], \
                          yerr=data['mag_list_err'][:,choose_region], fmt='ko', fillstyle='none' )
            plt.xlim(3,5)
            #plt.ylim(15,11.5)
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.xlabel('Log Wavelength (A)')
            plt.ylabel('AB Mag')
    
            # grab the model magnitudes for the best fit
            model_mag = np.empty(len(lambda_list))
            for m in range(len(lambda_list)):
                temp = np.array([ best_tau, best_av, 10**(best_log_age), best_bump, best_rv ])
                model_mag[m] = -2.5 * np.log10( grid_func[m](temp) * 10**best_log_mass ) - 48.6
                #temp = np.array([ best_tau, d14_av[choose_region], 10**(3.89), 0.4, 2 ])
                #model_mag[m] = -2.5 * np.log10( grid_func[m](temp) * 10**8.05 ) - 48.6
    
            print('best-fit model magnitudes:')
            print(model_mag)
    
            plt.plot(np.log10(lambda_list), model_mag, 'b^')
    
            #plt.tight_layout(h_pad=0.1)
            #plt.show()
            plt.savefig('./plots/spectrum_'+file_label[choose_region]+'.pdf')
            plt.close()
    
        #pdb.set_trace()
    
    
    return 





def rand_list(min_val, max_val, length):
    """
    Make a list of values uniformly randomly distributed between min_val and max_val.
    """

    return (max_val - min_val) * np.random.ranf(length) + min_val






def chi2_grid(grid_func, model_info, data):
    """
    This calculates the log probability for a particular set of model parameters.
    
    First, decide if the parameters are in the grid.  If they are, interpolate for
    the model magnitudes, then calculate log likelihood (-0.5 * chi2).

    grid_func - the list of functions from the grid interpolator
    model_info - a dictionary with the grid info
    data - a dictionary with the magnitudes/errors & fluxes/errors & assumed A_V
    """

    tau_list = np.linspace(model_info['tau_list'][0], 1000, num=10)
    age_list = np.geomspace( np.min(model_info['age_list']), np.max(model_info['age_list'])*0.999, num=11)
    #age_list = np.geomspace( 100., np.max(model_info['age_list'])*0.999, num=11)
    log_age_list = np.log10(age_list)
    bump_list = np.linspace(0, 2, num=11) #=7)
    #rv_list = model_info['rv_list']
    rv_list = np.linspace(2, 4.5, num=16) #=9)
    av_list = np.linspace(0.0, 1, num=6)

    length_tuple = ( len(tau_list), len(av_list), len(log_age_list), len(bump_list), len(rv_list) )
    
    new_chi2_grid = np.empty(length_tuple)
    new_mass_grid = np.empty(length_tuple)

    for index in np.ndindex(length_tuple):
    
        # grab the current parameters
        current_tau = tau_list[index[0]]
        current_av = av_list[index[1]]
        current_age = 10**(log_age_list[index[2]])
        current_bump = bump_list[index[3]]
        current_rv = rv_list[index[4]]

   
        # do grid interpolation
        model_flux = np.empty(len(data['mag_list']))
        for m in range(len(data['mag_list'])):
            temp = np.array([ current_tau, current_av, current_age, current_bump, current_rv ])
            model_flux[m] = grid_func[m](temp) #* 10**current_log_mass
        #model_mag = -2.5 * np.log10(model_flux) - 48.6


        # calculate a mass, weighted by errors
        new_mass_grid[index] = np.average(data['flux_list']/model_flux,
                                          weights=1/(data['flux_list_err']/model_flux) )
        
        # calculate chi2
        new_chi2_grid[index] = np.sum( (data['flux_list'] - new_mass_grid[index]*model_flux)**2 / data['flux_list_err']**2 )

    #pdb.set_trace()

    #print(datetime.datetime.now())
    
    # return log likelihood
    return {'new_chi2_grid':new_chi2_grid, 'new_mass_grid':np.log10(new_mass_grid),
                'grid_axes':{'tau_list':tau_list, 'av_list':av_list, 'log_age_list':log_age_list,
                                 'bump_list':bump_list, 'rv_list':rv_list},
                'axis_order':['tau_list','av_list','log_age_list','bump_list','rv_list']  }




if __name__ == '__main__':

    run_chi2()
