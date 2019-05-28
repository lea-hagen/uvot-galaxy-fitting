import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation
from scipy.optimize import curve_fit
from scipy.ndimage import label, gaussian_filter
from scipy.signal import argrelextrema, find_peaks

from best_val_mcmc import calc_best_vals
from plot_spec_mcmc import calc_model_mag

import pdb


def make_chains(file_label, burn_in, mstar_grid_func, two_pop,
                    remove_outliers=False, bimodal_check=None):
    """
    Read in the MCMC results and create chains
    * length for fitted parameters: n_walkers*n_steps
    * length for constant parameters: 1

    This will also create a chain for the stellar mass (length dependent
    on lengths of tau/age/mass)

    Parameters
    ----------
    file_label : string
        the label associated with the region/galaxy
    
    burn_in : int
        cut off this many steps at the beginning of each walker to account
        for burn-in

    mstar_grid_func : function
        function to convert tau + age into the fraction of mass that's stellar

    two_pop : dict or None
        if set, contains the dictionary with tau/log_age for a second population

    remove_outliers : boolean (default=False)
        if set, outliers will be removed, using the criteria of 
        information from:
        http://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564#11886564
        https://stats.stackexchange.com/questions/123895/mad-formula-for-outlier-detection

    bimodal_check : dict or None
        If set, contains the dictionary with 'mags'/'mag_err' (list of measured
        AB mags) and 'grid_func' (a list of functions, where each function
        outputs the model f_nu for each filter). Check the chi^2 of the best
        fits above/below a log_mass_ratio value, and pick the side with the best
        fit.

    Returns
    -------
    chains : dict
        dictionary of the chains for each parameter, and their MCMC log probabilities

    """



    # load the MCMC results
    pickle_file = open('./pickles/mc_'+file_label+'.pickle','rb')
    results = pickle.load(pickle_file)
    pickle_file.close()
    
    # get the info out
    sampler = results['sampler']
    fit_param = results['fit_param']
    n_fit = len(fit_param)
    const_param = results['const_param']
    two_pop = results['two_pop']

    # number of walkers/steps/fits
    n_walkers, n_steps, n_fit = sampler.chain.shape

    # make a plot of values vs. step number
    plt.figure(figsize=(9,8))
    # this takes a while for a lot of chains, so grab a random subset
    plot_these = np.random.randint(low=0, high=n_walkers, size=int(n_walkers/10.0))
    for i in range(n_fit):
        plt.subplot(n_fit, 1, i+1)  # n_rows, n_columns, plot_num
        plt.plot( np.transpose(sampler.chain[plot_these,:,i]), 'k-', alpha=0.1)
        plt.ylabel(fit_param[i])
    
    plt.subplots_adjust(0.08,0.05,0.94,0.97,0.2,0.2)
    plt.savefig('./plots/chains_'+file_label+'.pdf')
    plt.close()


    
    # cut off the burn-in period
    cut_chains = sampler.chain[:,burn_in:,:]
    cut_lnprob = sampler.lnprobability[:,burn_in:]

    # make a flat chain
    flatchain = np.reshape(cut_chains, (-1, n_fit))
    flat_lnprob = np.reshape(cut_lnprob, -1)

    # cut out some mass ratios
    if bimodal_check is not None and 'log_mass_ratio' in fit_param:
        ind = fit_param.index('log_mass_ratio')



        # fit two gaussians to the log_mass_ratio histogram
        # - make histogram
        hist_orig, bin_edges = np.histogram(flatchain[:,ind], density=True, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # - try smoothing
        hist = gaussian_filter(hist_orig, sigma=1)
        #hist_peaks = bin_centers[argrelextrema(hist, np.greater)]
        #hist_minima = bin_centers[argrelextrema(hist, np.less)]
        temp, _ = find_peaks(np.max(hist)-hist, prominence=(0.005,None))
        hist_minima = bin_centers[temp]
        # - find initial positions for gaussians
        mu1_init = bin_centers[hist == np.max(hist)][0]
        A1_init = np.max(hist)
        mu2_init = None
        i = 1
        while mu2_init == None:
            #pdb.set_trace()
            thresh = np.max(hist) - 0.01 * i
            if thresh <=0:
                mu2_init = mu1_init
                A2_init = 0.1
            above_thresh = (hist > thresh) & (bin_centers > -1)
            labeled_array, num_features = label(above_thresh)
            if num_features > 1:
                if np.sum(labeled_array == 1) < np.sum(labeled_array == 2):
                    mu2_init = bin_centers[labeled_array == 1][0]
                    A2_init = hist[labeled_array == 1][0]
                if np.sum(labeled_array == 1) > np.sum(labeled_array == 2):
                    mu2_init = bin_centers[labeled_array == 2][0]
                    A2_init = hist[labeled_array == 2][0]
            i += 1

        # - set initial conditions for first and second bumps
        if mu1_init < mu2_init:
            min_A = A1_init
            min_mu = mu1_init
            min_sigma = 0.35
            max_A = A2_init
            max_mu = mu2_init
            max_sigma = -0.25
        else:
            min_A = A2_init
            min_mu = mu2_init
            min_sigma = -0.25
            max_A = A1_init
            max_mu = mu1_init
            max_sigma = 0.35
            
        # - do the fitting
        try:
            coeff, var_matrix = curve_fit(gauss, bin_centers[bin_centers > -1], hist[bin_centers > -1],
                                            p0=[min_A,min_mu,min_sigma, max_A,max_mu,max_sigma],
                                              bounds=([-np.inf,-2.5,-np.inf,-np.inf,-2.5,-np.inf],
                                                          [np.inf,5.5,np.inf,np.inf,5.5,np.inf]))
        #  (in case of rare horrible failure:)
        except:
            coeff, var_matrix = curve_fit(gauss, bin_centers, hist,
                                            p0=[max_A/2,max_mu,max_sigma, max_A/2,max_mu,max_sigma])
        hist_fit = gauss(bin_centers, *coeff)
        A1, mu1, sigma1, A2, mu2, sigma2 = coeff

        # check if this is, in fact, bimodal
        bimodal = True
        if mu1 > mu2:
            A_hi = A1
            mu_hi = mu1
            sigma_hi = np.abs(sigma1)
            A_lo = A2
            mu_lo = mu2
            sigma_lo = np.abs(sigma2)
        if mu1 < mu2:
            A_hi = A2
            mu_hi = mu2
            sigma_hi = np.abs(sigma2)
            A_lo = A1
            mu_lo = mu1
            sigma_lo = np.abs(sigma1)
            
        if (sigma_lo > sigma_hi) and (mu_lo + 1.25*sigma_lo > mu_hi):
            bimodal = False
        if (sigma_hi > sigma_lo) and (mu_hi - 1.25*sigma_hi < mu_lo):
            bimodal = False
        if (A1 < 0.01) or (A2 < 0.01):
            bimodal = False
            
        # find location of the minimum between modes
        ind_list = np.where((bin_centers > mu_lo) & (bin_centers < mu_hi))[0]
        # if the list is empty, it's because both mu's are between two bins
        # -> set ind_list to the surrounding two points
        if len(ind_list) == 0:
            ind_list = np.array([ np.where(bin_centers < mu_hi)[0][-1] ])
        min_ind = ind_list[0]
        for i in ind_list:
            if hist_fit[i] < hist_fit[min_ind]:
                min_ind = i
        min_val = bin_centers[min_ind]
        #print('splitting chains at log_mass_ratio='+str(min_val))

        # verification plot
        plt.figure(figsize=(4,3))
        plt.plot(bin_centers, hist, marker='o', linestyle='None', ms=3, color='black')
        plt.plot(bin_centers, hist_fit, marker='.', linestyle='-', ms=3, color='red')
        plt.plot(bin_centers[min_ind], hist_fit[min_ind], marker='x', ms=5, color='xkcd:azure')
        plt.plot([-1,-1],[0,np.max(hist)], marker='None', linestyle=':', color='black', alpha=0.2)
        for peak in hist_minima:
            plt.plot([peak,peak], [0,np.max(hist)], marker='None', linestyle='--', color='blue', alpha=0.3)
        plt.annotate(r'A/$\mu$/$\sigma$', (0.03, 0.92), xycoords='axes fraction')
        plt.annotate('{0:.3f}, {1:.2f}, {2:.2f}'.format(A_lo, mu_lo, sigma_lo),
                         (0.03, 0.85), xycoords='axes fraction')
        plt.annotate('{0:.3f}, {1:.2f}, {2:.2f}'.format(A_hi, mu_hi, sigma_hi),
                         (0.03, 0.78), xycoords='axes fraction')
        if bimodal==False:
            plt.annotate('not bimodal!', (0.03, 0.7), xycoords='axes fraction')
        plt.xlabel('log_mass_ratio')
        plt.tight_layout()
        plt.savefig('./plots/fit_modes_'+file_label+'.pdf')
        plt.close()

        #pdb.set_trace()
        
        # only do the splitting/checking if the distribution is bimodal
        #if bimodal == True:
        if True:

            # boundaries of modes
            #if bimodal == True:
            #    mode_bounds = np.array([ np.min(flatchain[:,ind]), -1, min_val, np.max(flatchain[:,ind]) ])
            #else:
            #    mode_bounds = np.array([ np.min(flatchain[:,ind]), -1, np.max(flatchain[:,ind]) ])

            mode_bounds = np.concatenate(([np.min(flatchain[:,ind])], hist_minima, [np.max(flatchain[:,ind])]))
            
            chi2_list = np.full(len(mode_bounds)-1, np.nan)

            for b in range(len(mode_bounds)-1):
                # indices
                keep_ind = (flatchain[:,ind] >= mode_bounds[b]) & (flatchain[:,ind] < mode_bounds[b+1])
                # best fits
                best_fit = calc_best_vals(chain_to_dict(flatchain[keep_ind,:], fit_param, const_param))
                # corresponding model mags
                model_mag = calc_model_mag(bimodal_check['grid_func'], best_fit, two_pop=two_pop)
                # chi2
                chi2_list[b] = np.sum( (bimodal_check['mags'] - model_mag)**2 / bimodal_check['mag_err']**2 )
                
            # pick mode with lowest chi2
            best_chi2_ind = np.where(chi2_list == np.min(chi2_list))[0]
            keep_ind = (flatchain[:,ind] >= mode_bounds[best_chi2_ind]) & (flatchain[:,ind] < mode_bounds[best_chi2_ind+1])
            # grab those chains
            flatchain = flatchain[keep_ind,:]
            flat_lnprob = flat_lnprob[keep_ind]
 
                

        
    # remove outliers
    if remove_outliers == True and 'av' in fit_param:
        ind = fit_param.index('av')
        diff = np.abs(flatchain[:,ind] - np.median(flatchain[:,ind]))
        med_abs_deviation = np.median(diff)
        mod_z_score = 0.6745 * diff / med_abs_deviation
        bad = mod_z_score > 3.5
        flatchain = flatchain[~bad,:]
        flat_lnprob = flat_lnprob[~bad]
    if remove_outliers == True and 'rv' in fit_param:
        ind = fit_param.index('rv')
        diff = np.abs(flatchain[:,ind] - np.median(flatchain[:,ind]))
        med_abs_deviation = np.median(diff)
        mod_z_score = 0.6745 * diff / med_abs_deviation
        bad = mod_z_score > 3.5
        flatchain = flatchain[~bad,:]
        flat_lnprob = flat_lnprob[~bad]
       



    # save the main parameters

    chain = chain_to_dict(flatchain, fit_param, const_param)
    
            
    # calculate stellar mass

    max_length = np.max(np.array([ len(chain['tau']), len(chain['log_age']), len(chain['log_mass']) ]))

    if two_pop != None:
        max_length = np.max(np.array([ len(chain['tau']), len(chain['log_age']),
                                           len(chain['log_mass']), len(chain['log_mass_ratio']) ]))

    chain['log_st_mass'] = np.zeros(max_length)

    # save some extra info if two pops
    if two_pop != None:
        chain['log_st_mass_pop1'] = np.zeros(max_length)
        chain['log_st_mass_pop2'] = np.zeros(max_length)
        chain['log_st_mass_ratio'] = np.zeros(max_length)

    for i in range(max_length):
        # tau
        if len(chain['tau']) == 1:
            tau = chain['tau'][0]
        else:
            tau = chain['tau'][i]
        # log_age
        if len(chain['log_age']) == 1:
            age = 10**chain['log_age'][0]
        else:
            age = 10**chain['log_age'][i]
        # log_mass
        if len(chain['log_mass']) == 1:
            mass = 10**chain['log_mass'][0]
        else:
            mass = 10**chain['log_mass'][i]
        # log_mass_ratio
        if two_pop != None:
            if len(chain['log_mass_ratio']) == 1:
                mass_ratio = 10**chain['log_mass_ratio'][0]
            else:
                mass_ratio = 10**chain['log_mass_ratio'][i]
            
        # all together now
        pop1 = mstar_grid_func(np.array([tau, age])) * mass
        # - one pop
        if two_pop == None:
            chain['log_st_mass'][i] = np.log10(pop1)
        # - two pop
        if two_pop != None:
            pop2 = mstar_grid_func(np.array([two_pop['tau'], 10**two_pop['log_age']])) * mass * mass_ratio
            chain['log_st_mass'][i] = np.log10( pop1 + pop2 )
            chain['log_st_mass_pop1'][i] = np.log10(pop1)
            chain['log_st_mass_pop2'][i] = np.log10(pop2)
            chain['log_st_mass_ratio'][i] = np.log10(pop1/pop2)

            
    if remove_outliers == True and max_length > 1:
        diff = np.abs(chain['log_st_mass'] - np.median(chain['log_st_mass']))
        med_abs_deviation = np.median(diff)
        mod_z_score = 0.6745 * diff / med_abs_deviation
        bad = mod_z_score > 3.5
        for param in chain.keys():
            chain[param] = chain[param][~bad]
        flat_lnprob = flat_lnprob[~bad]
            
    # put the lnprob info into the chain dictionary too
    chain['lnprob'] = flat_lnprob


    # save the chains as a pickle
    pickle_file = open('./pickles/chains_'+file_label+'.pickle','wb')
    pickle.dump(chain, pickle_file)
    pickle_file.close()

    # return the chain info
    return chain
                


def chain_to_dict(flatchain, fit_param, const_param):
    """
    turn the chain into a dictionary

    Parameters
    ----------
    flatchain : array
        the flatchain in array form
        
    fit_param : list of strings
        list of parameters that are fit

    const_param : dict
        dictionary with any parameters held constant (and their values)

    Returns
    -------
    chain : dict
        the chains in dictionary form

    """

    chain = {}
    
    for param in const_param.keys():        
        # if the parameter was fit by emcee
        if param in fit_param:
            ind = fit_param.index(param)
            chain[param] = flatchain[:,ind]
        # if the parameter was held constant
        else:
            chain[param] = np.array([const_param[param]])


    return chain




def gauss(x, *p):
    """
    two gaussians
    """
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
