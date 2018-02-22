from __future__ import print_function
import numpy as np
import scipy
import scipy.io
import scipy.interpolate
#import cPickle as pickle  # (only 2.x)
import pickle
import time
#import is_outlier#; reload(is_outlier)
import os
import pdb
import importlib

import model_parameters

import best_val_chi2
importlib.reload(best_val_chi2)
import plot_triangle_chi2
importlib.reload(plot_triangle_chi2)
import plot_spec_chi2
importlib.reload(plot_spec_chi2)



def run_chi2(mag_list, mag_list_err, metallicity, distance, label,
                 re_run=False,
                 const_tau=-99, const_age=-99,
                 const_av=-99, const_rv=-99, const_bump=-99):
    """
    Run the modeling and create diagnostic plots


    Parameters
    ----------
    mag_list : dictionary
        dictionary with keys that are filter labels and values that are
        AB magnitudes

    mag_list_err : dictionary
        dictionary with keys that are filter labels and values that are
        uncertainties for the AB magnitudes

    metallicity : string
        one of ['005','01','015','02','025','035','050'], will be used to
        read in the proper model grid

    distance : float
        distance to the galaxy in Mpc

    label : string
        a label for the region/galaxy that will be used to construct the
        output file names

    re_run : boolean (default = False)
        if the fitting pickle file doesn't exist, the fitting will always
        proceed, but if the file does exist, you can choose whether to re-do
        the fitting part or just skip to the plotting

    const_tau, const_age, const_av, const_rv, const_bump : float (default = -99)
        if so inclined, hold a particular parameter constant at the specified
        value (the default of -99 means leave as free parameter)


    Returns
    -------
    nothing

    """

    print('')
    print('****************************')
    print('STARTING ', label)
    print('****************************')
    print('')

    setup_start = time.time()


    # -------------------------
    # set-up
    # -------------------------

    print('setting up the data...')

    # read in the pickle file with the model grid
    pickle_file = open('model_grid_'+metallicity+'.pickle','rb')
    model_info = pickle.load(pickle_file)
    pickle_file.close()
    # change tau list to numpy array
    model_info['tau_list'] = np.array(model_info['tau_list']).astype(np.float)
    # the wavelengths (angstroms) for each filter
    lambda_list_long = model_info['lambda_list']


    # input data
    # - list of filters, in order of increasing wavelength
    filter_list = [model_info['filter_list'][i] for i in np.argsort(lambda_list_long)
                       if model_info['filter_list'][i] in list(mag_list.keys()) ]
    n_filter = len(filter_list)
    # - wavelengths for each filter
    lambda_list = [lam for i,lam in enumerate(lambda_list_long)
                       if model_info['filter_list'][i] in filter_list]
    # - make an f_nu version
    fnu_list = np.zeros(n_filter)
    fnu_list_err = np.zeros(n_filter)
    for f in range(n_filter):
        fnu_list[f] = 10**( (mag_list[filter_list[f]] + 48.6)/(-2.5) )
        fnu_list_err[f] = fnu_list[f] / 2.5 * np.log(10) * mag_list_err[filter_list[f]]
    # - save magnitudes as np.array
    mag_list = np.array([mag_list[f] for f in filter_list])
    mag_list_err = np.array([mag_list_err[f] for f in filter_list])


    # print out the magnitudes
    print('')
    print('AB magnitudes:')
    for i in range(n_filter):
        print(filter_list[i], '{:.3f}'.format(mag_list[i]), '+/-',
              '{:.3f}'.format(mag_list_err[i]))
    print('')

    
    ## read in the IDL sav file with [adjusted] Draine+14 AVs
    #temp = scipy.io.readsav('../modeling_pix_av/d14_av_list.sav')
    #d14_av = temp['d14_av_list']
    ## divide the AVs in half
    #d14_av /= 2.0
    
        
    #print(data['mag_list'])
    #print(data['mag_list_err'])


    
    # generate the functions that will do grid interpolation for magnitudes
    grid_func = []
    for i in range(n_filter):
        # incorporate distance
        dist_pc = distance * 1e6
        # convert to f_nu
        model_fnu = 10**( (model_info['model_mags'][filter_list[i]]
                               - 2.5 * np.log10(10**2 / dist_pc**2) + 48.6)/(-2.5) )
        # interpolation function
        grid_func.append( scipy.interpolate.RegularGridInterpolator(( model_info['tau_list'], 
                                                                        model_info['av_list'], 
                                                                        model_info['age_list'], 
                                                                        model_info['bump_list'], 
                                                                        model_info['rv_list']), 
                                                                        model_fnu[:,:,:,:,:], 'linear', 
                                                                        bounds_error=True, fill_value=None) )
 

    # also make functions for stellar mass
    mstar_grid_func = scipy.interpolate.RegularGridInterpolator( (model_info['tau_list'], model_info['age_list']), 
                                                                      model_info['model_stellar_mass'], 'linear', 
                                                                      bounds_error=True, fill_value=None)

    
    #print(type(grid_func))


    # make sure sub-directories exist
    if not os.path.exists('./pickles/'):
        os.makedirs('./pickles/')
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

        
    #pdb.set_trace()

    # remove the giant grids from the model_info dictionary
    del model_info['model_mags']
    #del model_info['model_fnu']
    del model_info['model_stellar_mass']
    del model_info['readme']

    #pdb.set_trace()


    
    # -------------------------
    # run modeling
    # -------------------------

    modeling_start = time.time()

    # number of variables we're fitting (grid + mass)
    n_dimen = 5 + 1



    if (re_run == True) or (os.path.isfile('./pickles/chi2_'+label+'.pickle') == False):
        
        print('running chi2 fitter...')


        # the observed magnitudes/fluxes
        sub_data = {'mag_list':mag_list, 
                        'mag_list_err':mag_list_err, 
                        'flux_list':fnu_list, 
                        'flux_list_err':fnu_list_err}

        # holding things constant
        const_param = {'tau':const_tau, 'age':const_age,
                           'av':const_av, 'rv':const_rv, 'bump':const_bump}

        results = chi2_grid(grid_func, mstar_grid_func, model_info, sub_data, const_param)
        # results is dictionary with keys new_chi2_grid, new_mass_grid, grid_axes

        #pdb.set_trace()
            
        # save the chains from sampler object
        print('saving modeling results...')
        pickle_file = open('./pickles/chi2_'+label+'.pickle','wb')
        pickle.dump( results, pickle_file)
        pickle_file.close()

            
    modeling_end = time.time()
    
    
    # -------------------------
    # analyze the results
    # -------------------------

    print('analyzing results...')

    # best value
    best_fit = best_val_chi2.best_val(label)

    # triangle
    plot_triangle_chi2.triangle(label)

    # spectrum
    plot_spec_chi2.spectrum(lambda_list, mag_list, mag_list_err,
                                grid_func, label )


    print('')
    print('timing:')
    print('   set-up - ', modeling_start - setup_start, ' sec')
    print('   modeling - ', (modeling_end - modeling_start)/60, ' min')
    print('   plotting - ', time.time() - modeling_end, ' sec')
    print('')


    pdb.set_trace()


    
def chi2_grid(grid_func, mstar_grid_func, model_info, data, const_param):
    """
    This calculates the log probability for a particular set of model parameters.
    
    First, decide if the parameters are in the grid.  If they are, interpolate for
    the model magnitudes, then calculate log likelihood (-0.5 * chi2).

    grid_func - the list of functions from the grid interpolator
    mstar_grid_func - the list of functions from the stellar mass (M*) grid interpolator
    model_info - a dictionary with the grid info
    data - a dictionary with the magnitudes/errors & fluxes/errors
    const_param - dictionary with info about any parameters held constant
    """

    if const_param['tau'] == -99:
        tau_list = np.linspace(model_info['tau_list'][0], 1000, num=10)
    else:
        tau_list = np.array([const_param['tau']])
    
    if const_param['age'] == -99:
        age_list = np.geomspace( np.min(model_info['age_list']), np.max(model_info['age_list'])*0.999, num=11)
        #age_list = np.geomspace( 100., np.max(model_info['age_list'])*0.999, num=11)
        log_age_list = np.log10(age_list)
    else:
        age_list = np.array([const_param['age']])
        log_age_list = np.log10(age_list)

    if const_param['bump'] == -99:
        bump_list = np.linspace(0, 2, num=7) #=11)
    else:
        bump_list = np.array([const_param['bump']])

    if const_param['rv'] == -99:
        #rv_list = model_info['rv_list']
        rv_list = np.linspace(2, 4.5, num=9) #=16)
    else:
        rv_list = np.array([const_param['rv']])

    if const_param['av'] == -99:
        av_list = np.linspace(0.0, 1, num=6)
    else:
        av_list = np.array([const_param['av']])

        
    length_tuple = ( len(tau_list), len(av_list), len(log_age_list), len(bump_list), len(rv_list) )
    
    new_chi2_grid = np.zeros(length_tuple)
    new_mass_grid = np.zeros(length_tuple)
    new_st_mass_grid = np.zeros(length_tuple)

    n_filter = len(data['mag_list'])

    for index in np.ndindex(length_tuple):
    
        # grab the current parameters
        current_tau = tau_list[index[0]]
        current_av = av_list[index[1]]
        current_age = 10**(log_age_list[index[2]])
        current_bump = bump_list[index[3]]
        current_rv = rv_list[index[4]]

   
        # do grid interpolation
        model_flux = np.zeros(n_filter)
        temp = np.array([ current_tau, current_av, current_age, current_bump, current_rv ])
        for m in range(n_filter):
            model_flux[m] = grid_func[m](temp) #* 10**current_log_mass
            

        # calculate a mass, weighted by errors
        new_mass_grid[index] = np.average(data['flux_list']/model_flux,
                                          weights=1/(data['flux_list_err']/model_flux) )

        # calculate a stellar mass
        new_st_mass_grid[index] = new_mass_grid[index] * mstar_grid_func(np.array([ current_tau, current_age ]))
        
        # calculate chi2
        new_chi2_grid[index] = np.sum( (data['flux_list'] - new_mass_grid[index]*model_flux)**2 / data['flux_list_err']**2 )

    
    # return log likelihood
    return {'new_chi2_grid':new_chi2_grid,
                'new_mass_grid':np.log10(new_mass_grid),
                'new_st_mass_grid':np.log10(new_st_mass_grid),
                'grid_axes':{'tau_list':tau_list, 'av_list':av_list, 'log_age_list':log_age_list,
                                 'bump_list':bump_list, 'rv_list':rv_list},
                'axis_order':['tau_list','av_list','log_age_list','bump_list','rv_list']  }




if __name__ == '__main__':

    run_chi2()
