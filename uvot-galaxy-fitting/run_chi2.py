from __future__ import print_function
import numpy as np
import scipy
import scipy.io
import scipy.interpolate
#import cPickle as pickle  # (only 2.x)
import pickle
import datetime
import is_outlier#; reload(is_outlier)
import os.path
import pdb

import model_parameters

import best_val_chi2
import plot_triangle_chi2
import plot_spec_chi2

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

            results = chi2_grid(grid_func, mstar_grid_func, model_info, sub_data)
            # results is dictionary with keys new_chi2_grid, new_mass_grid, grid_axes

            #pdb.set_trace()
            
            # save the chains from sampler object
            print('saving chains...')
            pickle_file = open('./pickles/chi2_'+file_label[choose_region]+'.pickle','wb')
            pickle.dump( results, pickle_file)
            pickle_file.close()

            
        print('END TIME:')
        print(datetime.datetime.now())
        print('')
    
    
        # -------------------------
        # analyze the results
        # -------------------------

        print('analyzing results...')

        # best value
        best_val_chi2.best_val(file_label[choose_region])

        # triangle
        plot_triangle_chi2.triangle(file_label[choose_region])

        # spectrum
        plot_spec_chi2.spectrum(lambda_list,
                                    data['mag_list'][:,choose_region],
                                    data['mag_list_err'][:,choose_region],
                                    grid_func, file_label[choose_region] )

    
    return 






def chi2_grid(grid_func, mstar_grid_func, model_info, data):
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
    new_st_mass_grid = np.empty(length_tuple)

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

        # calculate a stellar mass
        new_st_mass_grid[index] = new_mass_grid[index] * mstar_grid_func(np.array([ current_tau, current_age ]))
        
        # calculate chi2
        new_chi2_grid[index] = np.sum( (data['flux_list'] - new_mass_grid[index]*model_flux)**2 / data['flux_list_err']**2 )

    #pdb.set_trace()

    #print(datetime.datetime.now())
    
    # return log likelihood
    return {'new_chi2_grid':new_chi2_grid,
                'new_mass_grid':np.log10(new_mass_grid),
                'new_st_mass_grid':np.log10(new_st_mass_grid),
                'grid_axes':{'tau_list':tau_list, 'av_list':av_list, 'log_age_list':log_age_list,
                                 'bump_list':bump_list, 'rv_list':rv_list},
                'axis_order':['tau_list','av_list','log_age_list','bump_list','rv_list']  }




if __name__ == '__main__':

    run_chi2()
