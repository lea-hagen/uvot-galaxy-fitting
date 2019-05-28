import numpy as np
import pickle
import pdb


def best_val(file_label, chain, verbose=True):
    """
    Read in the MCMC results file and extract the best fit

    This will both
        * return the best fit values/uncertainties
        * write them out to a file

    Parameters
    ----------
    file_label : string
        the label associated with the region/galaxy
    
    chain : dict
        the trimmed and flattened chains (length N for fitted, length 1 for
        held constant), output from make_chains_mcmc.py

    verbose : boolean
        set to True to print out best fits

    Returns
    -------
    best_fits : dict
        dictionary of the best fits for each parameter

    """



    # a file to save the results
    results_file = open('./results/results_'+str(file_label)+'.list','w')
    results_file.write('#  value  upper_uncertainty   lower_uncertainty   (errors are -99 for parameters held constant) \n')

    #pdb.set_trace()
    # if 'log_mass_ratio' not in chain.keys():
    #     param_list = ['tau','av','log_age','bump','rv','log_mass','log_st_mass']
    #     results_file.write('#   tau                      av                    log_age               bump                  rv                    log_mass              log_stellar_mass \n')
    # else:
    #     param_list = ['tau','av','log_age','bump','rv','log_mass','log_mass_ratio','log_st_mass']
    #     results_file.write('#   tau                      av                    log_age               bump                  rv                    log_mass              log_mass_ratio        log_stellar_mass \n')


    # get list of parameters
    param_list = chain.keys()
    # preferred order of keys in the output file
    param_order = ['tau','av','log_age','bump','rv','log_mass','log_mass_ratio','log_st_mass',
                       'log_st_mass_pop1','log_st_mass_pop2','log_st_mass_ratio']
    # header for results file
    file_header = '#   '
    for par in param_order:
        if par in param_list:
            if par == 'tau':
                file_header += '{:25}'.format(par)
            else:
                file_header += '{:22}'.format(par)
    file_header += ' \n'
    results_file.write(file_header)
                
        
    # find the best fits
    best_fit = calc_best_vals(chain)
    

    # write/print it all out
    if verbose == True:
        print('best fits:')
    for param in param_order:
        if param in param_list:
            results_file.write(str(best_fit[param]) + ' ' +
                                str(best_fit[param+'_hi_err']) + ' ' +
                                str(best_fit[param+'_lo_err']) + '  ')
            if verbose == True:
                print(param + ':  ' + '{0:.4f}'.format(best_fit[param])
                        + ' +' + '{0:.4f}'.format(best_fit[param+'_hi_err'])
                        + ' -' + '{0:.4f}'.format(best_fit[param+'_lo_err']) )

        

    # close the file
    results_file.write('\n')
    results_file.close()


    return best_fit



def calc_best_vals(chain):
    """
    The calculations of the best fits

    Parameters
    ----------
    chain : dict
        the trimmed and flattened chains (length N for fitted, length 1 for
        held constant), output from make_chains_mcmc.py

    Returns
    -------
    best_fit : dict
        dictionary of the best fits for each parameter
 
    """

    # a dictionary to save the best fits
    best_fit = {}
    
    # calculate best fits
   
    for param in chain.keys():
        
        # if the parameter was fit by emcee
        if len(chain[param]) > 1:
            best_val, hi_err, lo_err = get_vals(chain[param])
            best_fit[param] = float(best_val)
            best_fit[param+'_hi_err'] = float(hi_err)
            best_fit[param+'_lo_err'] = float(lo_err)
        # if the parameter was held constant
        else:
            best_fit[param] = float('{0:.4f}'.format(chain[param][0]))
            best_fit[param+'_hi_err'] = -99
            best_fit[param+'_lo_err'] = -99

    return best_fit

    
def get_vals(array):
    """
    calculate the best value and uncertainties using the 16th/50th/84th
    percentiles, and print them out nicely
    """

    p16, p50, p84 = np.percentile(array, [16, 50, 84])

    best_val = '{0:.4f}'.format(p50)
    lo_err = '{0:.4f}'.format(p50 - p16)
    hi_err = '{0:.4f}'.format(p84 - p50)

    return best_val, hi_err, lo_err
