from __future__ import print_function
import numpy as np
import scipy
import scipy.io
import scipy.interpolate
import emcee
#import cPickle as pickle  # (only 2.x)
import pickle
import time
#import is_outlier#; reload(is_outlier)
import os
import pdb
import importlib

import model_parameters

import make_chains_mcmc
importlib.reload(make_chains_mcmc)
import best_val_mcmc
importlib.reload(best_val_mcmc)
import plot_triangle_mcmc
importlib.reload(plot_triangle_mcmc)
import plot_spec_mcmc
importlib.reload(plot_spec_mcmc)

import pdb


def run_mcmc(mag_list, mag_list_err, metallicity, distance, label, dust_geom='screen',
                 n_walkers=2000, n_steps=2000, burn_in=800,
                 re_run=False,
                 const_tau=None, const_age=None,
                 const_av=None, const_rv=None, const_bump=None,
                 const_mass=None, const_mass_ratio=None,
                 two_pop=None,
                 remove_outliers=False):
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

    dust_geom : string (default='screen')
        If set to 'screen', dust will be in a sheet in front of the whole
        stellar population.  If set to 'disk', the dust will be in an
        infinitely thin disk with half the stars in front and half behind.  

    n_walkers : int (default = 2000)
        number of MCMC walkers 

    n_steps : int (default = 2000)
        number of steps for each walker

    burn_in : int (default = 800)
        cut off this many steps at the beginning of each walker to account
        for burn-in

    re_run : boolean (default = False)
        if the fitting pickle file doesn't exist, the fitting will always
        proceed, but if the file does exist, you can choose whether to re-do
        the fitting part or just skip to the plotting

    const_tau, const_age, const_av, const_rv, const_bump, const_mass, const_mass_ratio : float (default = None)
        if so inclined, hold a particular parameter constant at the specified
        value (the default of None means leave as free parameter).  Note that
        age/mass/mass_ratio are LOG10 values (e.g., 10^3 Myr should be entered
        as 3).  const_mass_ratio will be ignored unless two_pop is set.

    two_pop : dict or None (default=None)
       If you want to have add a second stellar population, put it here.
       Should be a dictionary with keys of 'tau' and 'log_age' (both in Myr).

    remove_outliers : boolean (default=False)
       If this is set to True, the chain for A_V will be used to find outliers,
       defined using the MAD (see make_chains_mcmc.py for more details)


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
    if dust_geom == 'screen':
        pickle_file = open('model_grid_'+metallicity+'_screen.pickle','rb')
    if dust_geom == 'disk':
        pickle_file = open('model_grid_'+metallicity+'_disk.pickle','rb')
    if (dust_geom != 'screen') and (dust_geom != 'disk'):
        print('run_mcmc: must choose screen or disk!')
        return
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
    lambda_list = [lambda_list_long[i] for i in np.argsort(lambda_list_long)
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


    # holding things constant
    const_param = {'tau':const_tau, 'log_age':const_age, 'av':const_av,
                       'rv':const_rv, 'bump':const_bump, 'log_mass':const_mass}
    if two_pop != None:
        const_param['log_mass_ratio'] = const_mass_ratio


    # make sure sub-directories exist
    if not os.path.exists('./pickles/'):
        os.makedirs('./pickles/')
    if not os.path.exists('./plots/'):
        os.makedirs('./plots/')
    if not os.path.exists('./results/'):
        os.makedirs('./results/')


    # -------------------------
    # run emcee
    # -------------------------

    modeling_start = time.time()

    # max number of variables we're fitting (grid + mass)
    n_dimen = 5 + 1
    # and also possibly the mass ratio
    if two_pop != None:
        n_dimen += 1

    # list of parameters we're fitting
    all_param = ['tau','av','log_age','bump','rv','log_mass']
    if two_pop != None:
        all_param.append('log_mass_ratio')
    fit_param = [p for p in all_param if const_param[p] == None]
    n_fit = len(fit_param)
    
        
    # set up starting positions for each walker:
    # randomly distributed in parameter space
    
    # format -- a list (length n_walkers) of arrays (each length n_fit)
    
    init_pos = []
    
    for i in range(n_walkers):

        temp = []
        if const_param['tau'] == None:
            temp.append(np.random.uniform(low=np.min(model_info['tau_list']),
                                            high=np.max(model_info['tau_list'])) )
        if const_param['av'] == None:
            temp.append(np.random.uniform(low=np.min(model_info['av_list']),
                                            high=np.max(model_info['av_list'])) )
        if const_param['log_age'] == None:
            temp.append(np.random.uniform(low=np.log10(np.min(model_info['age_list'])),
                                            high=np.log10(np.max(model_info['age_list'])) ))
        if const_param['bump'] == None:
            temp.append(np.random.uniform(low=np.min(model_info['bump_list']),
                                            high=np.max(model_info['bump_list'])) )
        if const_param['rv'] == None:
            temp.append(np.random.uniform(low=np.min(model_info['rv_list']),
                                            high=np.max(model_info['rv_list'])) )
        if const_param['log_mass'] == None:
            temp.append(np.random.uniform(low=6, high=8) )

        if two_pop != None:
            if const_param['log_mass_ratio'] == None:
                temp.append(np.random.uniform(low=-2, high=3) )
        
        init_pos.append( np.array(temp) )


    # do the running


    if (re_run == True) or (os.path.isfile('./pickles/mc_'+label+'.pickle') == False):
        
        print('running mcmc fitter...')


        # the observed magnitudes/fluxes
        obs_phot = {'mag_list':mag_list, 
                        'mag_list_err':mag_list_err, 
                        'flux_list':fnu_list, 
                        'flux_list_err':fnu_list_err}
            
        # emcee
        # - initialize sampler
        sampler = emcee.EnsembleSampler(n_walkers, n_fit, lnprob,
                                            args=(grid_func, model_info, obs_phot, fit_param, const_param, two_pop))
        # - run modeling
        sampler.run_mcmc(init_pos, n_steps)

        


        #pdb.set_trace()
            
        # save the chains from sampler object
        print('saving modeling results...')
        pickle_file = open('./pickles/mc_'+label+'.pickle','wb')
        results = {'sampler':sampler, 'fit_param':fit_param, 'const_param':const_param, 'two_pop':two_pop}
        pickle.dump(results, pickle_file)
        pickle_file.close()

            
    modeling_end = time.time()


    # -------------------------
    # analyze the results
    # -------------------------

    print('analyzing results...')


    # extract chains
    chains = make_chains_mcmc.make_chains(label, burn_in, mstar_grid_func, two_pop,
                                              remove_outliers=remove_outliers)
    
    # best value
    best_fit = best_val_mcmc.best_val(label, chains)

    # triangle
    plot_triangle_mcmc.plot_triangle(label, chains)

    # spectrum
    plot_spec_mcmc.spectrum(lambda_list, mag_list, mag_list_err,
                                grid_func, best_fit, label, two_pop )


    print('')
    print('timing:')
    print('   set-up - ', modeling_start - setup_start, ' sec')
    print('   modeling - ', (modeling_end - modeling_start)/60, ' min')
    print('   plotting - ', time.time() - modeling_end, ' sec')
    print('')


def lnprob(theta, grid_func, model_info, data, fit_param, const_param, two_pop):
    """
    This calculates the log probability for a particular set of model parameters.
    
    First, decide if the parameters are in the grid.  If they are, interpolate for
    the model magnitudes, then calculate log likelihood (-0.5 * chi2).


    Parameters
    ----------

    theta : array
        an array of some combination of r_v, a_v, bump strength, age, mass (from emcee)

    grid_func : list
        the list of functions from the grid interpolator

    model_info : dict
        a dictionary with the grid info

    data : dict
        dictionary with the magnitudes/errors & fluxes/errors, labeled as
        'mag_list', 'mag_list_err', 'flux_list', 'flux_list_err'

    fit_param : list
        list of string names for the parameters we're fitting

    const_param : dict
        dictionary with info about any parameters held constant

    two_pop : dict or None
        if set, the tau/age for a second population (see run_mcmc docstring)


    """

    # grab the current parameters
    # - the values held constant
    current_val = {key:const_param[key] for key in const_param.keys() if const_param[key] != None}
    # - the values fit by emcee
    for i in range(len(fit_param)):
        current_val[fit_param[i]] = theta[i]


    # check that they're within the grid... if not, return -infinity
    if current_val['tau'] < np.min(model_info['tau_list']) \
        or current_val['tau'] > np.max(model_info['tau_list']) \
        or current_val['av'] < np.min(model_info['av_list']) \
        or current_val['av'] > np.max(model_info['av_list']) \
        or 10**current_val['log_age'] < np.min(model_info['age_list']) \
        or 10**current_val['log_age'] > np.max(model_info['age_list']) \
        or current_val['bump'] < np.min(model_info['bump_list']) \
        or current_val['bump'] > np.max(model_info['bump_list']) \
        or current_val['rv'] < np.min(model_info['rv_list']) \
        or current_val['rv'] > np.max(model_info['rv_list']):
        return -np.inf
    if two_pop != None:
        if current_val['log_mass_ratio'] < -2 or current_val['log_mass_ratio'] > 5:
            return -np.inf
    
    # do grid interpolation
    model_flux = np.zeros(len(data['mag_list']))
    
    for m in range(len(data['mag_list'])):
        
        temp = np.array([ current_val['tau'], current_val['av'], 10**(current_val['log_age']),
                              current_val['bump'], current_val['rv'] ])
        model_flux[m] = grid_func[m](temp) * 10**current_val['log_mass']

        # possibly do another constant population too
        if two_pop != None:
            temp = np.array([two_pop['tau'], current_val['av'], 10**two_pop['log_age'],
                                 current_val['bump'], current_val['rv'] ])
            model_flux_2 = grid_func[m](temp) * 10**current_val['log_mass'] * 10**current_val['log_mass_ratio']
            model_flux[m] += model_flux_2
        
    #model_mag = -2.5 * np.log10(model_flux) - 48.6


    # calculate chi2
    # -> remove NaNs and 0 fluxes (the data>0 works to filter nans too)
    use = [data['flux_list'] > 0.0] 
    #chi2 = np.sum( (data['mag_list'] - model_mag)**2 / data['mag_list_err']**2 )
    chi2 = np.sum( (data['flux_list'][use] - model_flux[use])**2 / data['flux_list_err'][use]**2 )


    #print(datetime.datetime.now())
    
    # return log likelihood
    return -0.5 * chi2
