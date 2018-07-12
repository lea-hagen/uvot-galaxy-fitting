import numpy as np
import pickle
import matplotlib.pyplot as plt
import pdb


def make_chains(file_label, burn_in, mstar_grid_func, two_pop):
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

    Returns
    -------
    chains : dict
        dictionary of the chains for each parameter

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
        plt.plot( np.transpose(sampler.chain[plot_these,:,i]), 'k-')
        plt.ylabel(fit_param[i])
    
    plt.subplots_adjust(0.08,0.05,0.94,0.97,0.2,0.2)
    plt.savefig('./plots/chains_'+file_label+'.pdf')
    plt.close()


    
    # cut off the burn-in period
    cut_chains = sampler.chain[:,burn_in:,:]

    # make a flat chain
    flatchain = np.reshape(cut_chains, (-1, n_fit))


    # save the main parameters

    chain = {}
    
    for param in const_param.keys():        
        # if the parameter was fit by emcee
        if param in fit_param:
            ind = fit_param.index(param)
            chain[param] = flatchain[:,ind]
        # if the parameter was held constant
        else:
            chain[param] = np.array([const_param[param]])

            
    # calculate stellar mass

    max_length = np.max(np.array([ len(chain['tau']), len(chain['log_age']), len(chain['log_mass']) ]))

    if two_pop != None:
        max_length = np.max(np.array([ len(chain['tau']), len(chain['log_age']),
                                           len(chain['log_mass']), len(chain['log_mass_ratio']) ]))

    chain['log_st_mass'] = np.zeros(max_length)

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

        
    # return the chain info
    return chain
                
