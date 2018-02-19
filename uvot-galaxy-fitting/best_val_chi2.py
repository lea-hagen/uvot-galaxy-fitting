import numpy as np
import pickle


def best_val(file_label, verbose=True):
    """
    Read in the chi2 results grid file and grab the best fit

    This will both
        * return the best fit values
        * write them out to a file

    Parameters
    ----------
    file_label : string
        the label associated with the region/galaxy
    
    verbose : boolean
        set to True to print out best fits

    Returns
    -------
    best_fits : dict
        dictionary of the best fits for each parameter

    """


    # load the chi2 grid
    pickle_file = open('./pickles/chi2_'+file_label+'.pickle','rb')
    results = pickle.load(pickle_file)
    pickle_file.close()


    # number of dimensions to fit
    n_dimen = len(results['axis_order'])
    
    # array to hold results
    best_val = np.empty(n_dimen)
        
    # a file to save the results
    results_file = open('./results/results_'+file_label+'.list','w')
    results_file.write('#   tau       av    log_age   bump    rv     log_mass    log_stellar_mass \n')

    # get the indices for the best chi2
    best_chi2_index = np.where(results['new_chi2_grid'] == np.min(results['new_chi2_grid']))
        
    for i in range(n_dimen):

        # calculating
        current_axis = results['axis_order'][i]
        best_val[i] = results['grid_axes'][current_axis][best_chi2_index[i][0]]
        #print('current_axis: ', current_axis)
        #print('   Best value: ', '{:.3f}'.format(best_val[i]))
        results_file.write('   ' + '{:.3f}'.format(best_val[i]))

    # get named values
    best_tau, best_av, best_log_age, best_bump, best_rv  = best_val

    # corresponding best mass and stellar mass
    best_log_mass = results['new_mass_grid'][best_chi2_index][0]
    best_log_st_mass = results['new_st_mass_grid'][best_chi2_index][0]
    results_file.write('   ' + '{:.3f}'.format(best_log_mass) )
    results_file.write('   ' + '{:.3f}'.format(best_log_st_mass) )

                  
    # print it all out
    if verbose == True:
        print('   best tau, av, log_age, bump, rv, log_mass, log_stellar_mass:')
        print('   ', best_val, '  ', best_log_mass, '  ', best_log_st_mass)
        print('')

        
    # close the file
    results_file.write('\n')
    results_file.close()

    
    # return best fits
    return {'tau':best_tau, 'av':best_av, 'log_age':best_log_age,
                'bump':best_bump, 'rv':best_rv,
                'log_mass':best_log_mass, 'log_st_mass':best_log_st_mass,
                'best_chi2_index':best_chi2_index}
