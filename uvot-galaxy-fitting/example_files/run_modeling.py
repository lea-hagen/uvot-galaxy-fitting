import numpy as np
import pickle

import pdb

import importlib
import model_parameters

import create_grids
importlib.reload(create_grids)
import run_chi2
importlib.reload(run_chi2)

def run_modeling():
    """
    Wrapper to do the modeling
    """


    # ---- grid-related things

    # the list of filters to put in the grid
    # -> these will be used as the labels
    filter_list_grid = ['fuv','nuv','w2','m2','w1','u','g','r','i','3.6']

    # the full path to the corresponding filter transmission curves
    temp = ['galex_fuv.txt','galex_nuv.txt', 'swift_w2.txt','swift_m2.txt','swift_w1.txt',
                'sdss_u.txt','sdss_g.txt','sdss_r.txt','sdss_i.txt','irac_3.6.dat']
    filter_transmission_curves = ['/astro/dust_kg3/lhagen/swift/filters/' + i for i in temp]


    # metallicity: choose from ['005','01','015','02','025','035','050']
    metallicity = '02'

    
    # create grids
    create_grids.create_grids(filter_list_grid, filter_transmission_curves, metallicity)

    
    # ---- modeling-related things

    # filters to model
    filter_list_data = ['fuv','nuv','w2','m2','w1','g','r','i','3.6']

    # ... do your procedure to read in the data ...

    # ... but here's some example photometry so it'll run
    phot = np.array([16.64, 16.27, 16.49, 16.32, 16.23, 14.75, 14.41, 14.17, 14.78])
    phot_err = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15])
    
    # number of galaxies
    n_gal = 1

    # distance
    dist = 9.0 # Mpc

    # run modeling
    for g in range(n_gal):
        mags = {f:phot[i] for i,f in enumerate(filter_list_data)}
        mag_errs = {f:phot_err[i] for i,f in enumerate(filter_list_data)}
        run_chi2.run_chi2(mags, mag_errs, metallicity, dist, 'UGC00695')

    

    


if __name__ == '__main__':

    run_modeling()
