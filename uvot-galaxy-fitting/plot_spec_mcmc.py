import numpy as np
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
import pickle

import model_parameters
import best_val_chi2

import pdb

def spectrum(lambda_list, mag, mag_err, grid_func,
                 best_fit, file_label, two_pop,
                 best=False, verbose=False):
    """
    Make a spectrum of the data and the best-fit model, and save the model magnitudes

    The best-fit parameters are extracted using the file_label with best_val_chi2.py


    Parameters
    ----------
    lambda_list : array of floats
        wavelengths (in Angstroms) for each photometric point

    mag : array of floats
        AB magnitudes of the photometry

    mag_err : array of floats
        errors for each magnitude

    grid_func : list of functions
        a list of functions, where each function outputs the model f_nu for each filter

    best_fit : dict
        the dictionary with the best-fit values (and errors) output from best_val_mcmc.py

    file_label : string
        the label associated with the region/galaxy

    two_pop : dict or None
        if set, contains the dictionary with tau/log_age for a second population
        
    best : boolean (default=False)
        set to True to use "best" fit (rather than 50th percentile) to create model SED

    verbose : boolean
        set to True to print out model magnitudes

    Returns
    -------
    nothing

    """

    # plot photometry
    plt.figure(figsize=(7,5))
    plt.errorbar( np.log10(lambda_list), mag, \
                  yerr=mag_err, fmt='ko', fillstyle='none' )
    plt.xlim(3,5)
    #plt.ylim(15,11.5)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.xlabel('Log Wavelength (A)')
    plt.ylabel('AB Mag')

    # file to save the model magnitudes
    model_file = open('./results/modelmag_'+str(file_label)+'.list','w')
    model_file.write('# wavelength (A)     model (AB mag) \n')
   
    
    # grab the model magnitudes for the best fit

    # set which parameter to use
    b_str = ''
    if best:
        b_str = '_best'

    # calculate model magnitudes
    params = set([key.replace('_lo_err','').replace('_hi_err','').replace('_best','')
                                        for key in best_fit.keys()])
    input_dict = {p:best_fit[p+b_str] for p in params}
    model_mag = calc_model_mag(grid_func, input_dict, two_pop=two_pop)
    
    # write it out
    [model_file.write(' ' + str(lambda_list[m]) + '      ' + str(model_mag[m]) + '\n')
         for m in range(len(model_mag)) ]

        
    # close file
    model_file.close()

    if verbose == True:
        print('best-fit model magnitudes:')
        print(model_mag)

    plt.plot(np.log10(lambda_list), model_mag, 'b^')

    #plt.tight_layout(h_pad=0.1)
    #plt.show()
    plt.savefig('./plots/spectrum_'+file_label+'.pdf')
    plt.close()




def calc_model_mag(grid_func, model_val, two_pop=None):
    """
    The math for going from model parameters -> model magnitudes
    

    Parameters
    ----------
    grid_func : list of functions
        a list of functions, where each function outputs the model f_nu for each filter

    model_val : dict
        dictionary with the values at which to evaluate the model grid: tau, av, log_age, bump, rv, log_mass, and possibly log_mass_ratio (used only if two_pop is set)

    two_pop : dict or None
        if set, contains the dictionary with tau/log_age for a second population

    Returns
    -------
    model_mag : array of floats
        the model magnitudes

    """

    model_mag = np.zeros(len(grid_func))
    for m in range(len(grid_func)):

        temp = np.array([ model_val['tau'], model_val['av'], 10**model_val['log_age'],
                            model_val['bump'], model_val['rv'] ])
        model_flux = grid_func[m](temp) * 10**model_val['log_mass']
        
        # possibly do another constant population too
        if two_pop != None:
            temp = np.array([two_pop['tau'], model_val['av'], 10**two_pop['log_age'],
                                model_val['bump'], model_val['rv'] ])
            model_flux_2 = grid_func[m](temp) * 10**model_val['log_mass'] * 10**model_val['log_mass_ratio']
            model_flux += model_flux_2

        # flux -> AB mag
        model_mag[m] = -2.5 * np.log10(model_flux) - 48.6

    
    return model_mag
