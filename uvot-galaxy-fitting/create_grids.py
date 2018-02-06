import numpy as np
import pysynphot as S
import matplotlib.pyplot as plt
import pickle
import time
import math

from config import __ROOT__

import dust_c10
import model_parameters
import spec_filter


import pdb

def create_grids(filter_label, filter_file, metallicity):
    """
    Make a large grid with these axes:
    * tau
    * av
    * age
    * bump
    * rv

    For a given metallicity, if the file doesn't exist, this will create a
    file and populate the grid with models in all filters.  If the file does
    exist, it will append any missing filters.


    Parameters
    ----------
    filter_label : list of strings
        String label(s) associated with each filter passband.  Any photometry
        for each of these filters must have the same matching label for the
        given passband.

    filter_file : list of strings
        Full path+file name for the transmission function for each filter.
        Files be organized as two columns: wavelength (Angstroms) and
        transmission.

    metallicity : string
       one of ['005','01','015','02','025','035','050']


    Returns
    -------
    file_path : string
       full path to the PEGASE model file
    
    """


    # list of metallicities
    metallicity_list = ['005','01','015','02','025','035','050']
    # match to the chosen metallicity
    choose_metallicity = metallicity_list.index(metallicity)

    # list of star formation histories
    #   b = burst
    #   #### = tau (Myr) for exponentially declining SFH
    #   const = constant SFH
    #   g### = tau (Myr) for exponentially increasing SFH
    sfh = ['b','0110','0140','0155','0170','0185','0200','0215','0235','0250','0275', 
            '0300','0350','0400','0500','0600','0750','1000','1500','3500', 
            'const','g020','g040','g075','g150']
    # for now just use exponentially declining
    tau_list = sfh[1:20]

    # extract grid values from model_parameters
    bump_list = model_parameters.bump_list
    rv_list = model_parameters.rv_list
    av_list = model_parameters.av_list


    # read in one spectrum to get some of the grid info
    spec_lambda, age_list, spec, _, _, _ = writespec_info(pegase_file(__ROOT__+'/pegase_grids/', '005','b'))

    # limit the age to 1 Myr to 13 Gyr
    age_subset = np.where((age_list >= 1) & (age_list <= 13000))[0]
    age_list = age_list[age_subset]

    # some testing
    fuv = np.loadtxt(filter_files[0])
    w = fuv[:,0]
    t = fuv[:,1]
    bp = S.ArrayBandpass(w, t, name='FUV')
    sp = S.ArraySpectrum(spec_lambda, spec[5,:], name='Model', fluxunits='flam')
    obs = S.Observation(sp, bp)

    # make a list of pysynphot bandpasses and their avg wavelengths
    # also plain lists of transmission info
    bp_list = []
    bp_lambda = []
    bp_trans = []
    lambda_list = np.array([])
    for filt in filter_files:
        bp_table = np.loadtxt(filt)
        new_bp = S.ArrayBandpass(bp_table[:,0], bp_table[:,1])
        bp_list.append(new_bp)
        bp_lambda.append(bp_table[:,0])
        bp_trans.append(bp_table[:,1])
        lambda_list = np.append(lambda_list, new_bp.avgwave())
    

    # figure out the largest wavelength from the model spectrum to save
    #max_lambda = np.max(np.array( [np.max( np.loadtxt(model_parameters.filter_transmission_curves[i])[:,0]) for i in range(len(model_parameters.filter_list))] ))
    
    # initialize a giant array to hold the grid of model magnitudes
    temp_array = np.zeros(( len(tau_list), len(av_list), len(age_list), len(bump_list), len(rv_list) ))
    model_mags = {}
    for f in filter_label:
        model_mags[f] = temp_array

    # initialize arrays to hold the grid of masses
    model_stellar_mass = np.zeros(( len(tau_list), len(age_list) ))
    model_remnant_mass = np.zeros(( len(tau_list), len(age_list) ))
    model_gas_mass = np.zeros(( len(tau_list), len(age_list) ))

    # set a distance of 10 pc
    dist = 3.08567758e19

    
    # now go populate the array

    for t in range(len(tau_list)):
    #for t in [0]:

        tau_start = time.time()
        print('tau item ', str(t), ' of ', str(len(tau_list)))

        # read in the spectra
        filename = pegase_file(__ROOT__+'/pegase_grids/', model_parameters.metallicity, tau_list[t] )
        _, _, spec, m_stellar, m_remnant, m_gas = writespec_info(filename)

        # put spectra at the chosen distance (erg/s/A -> erg/s/cm^2/A)
        spec = spec / (4 * math.pi * dist^2)


        # save the mass info
        model_stellar_mass[t,:] = m_stellar[age_subset]
        model_remnant_mass[t,:] = m_remnant[age_subset]
        model_gas_mass[t,:] = m_gas[age_subset]

        # apply dust to each spectrum
        for index in np.ndindex(( len(av_list), len(rv_list), len(bump_list) )):

            #print('    ', index)

            # create an extinction curve
            ext = dust_c10.dust_c10(av_list[index[0]], rv_list[index[1]], bump_list[index[2]], spec_lambda)
            # put it in flux units
            ext_flux = 10**(-0.4 * ext)
             
            # apply it to each age slice of the spectrum
            for a in range(len(age_list)):
                #spec_slice = S.ArraySpectrum(spec_lambda, spec[a,:] * ext_flux, fluxunits='flam')
                spec_slice = spec[a,:] * ext_flux
                
                # which requires putting the spectrum through each bandpass
                for f in range(len(filter_label)):
                    #mag = S.Observation(spec_slice, bp_list[f]).effstim('ABMag')
                    #mag = S.Observation(spec_slice, bp_list[f], binset=spec_slice.wave).effstim('ABMag')
                    mag = -2.5 * np.log10(spec_filter.spec_filter(spec_lambda, spec_slice, bp_lambda[f], bp_trans[f])) - 48.6
                    model_mags[filter_label[f]][t, index[0], a, index[2], index[1]] = mag
                        
                #pdb.set_trace()  
                #temp = model_mags[t, index[0], a, index[2], index[1], :]
                #spec_slice.convert('ABMag')
                #plt.plot(np.log10(spec_slice.wave), spec_slice.flux)
                #plt.plot(np.log10(lambda_list), temp, marker='o')
                #plt.ylim(np.max(temp)+0.2, np.min(temp)-0.2)
                #plt.show()
                #pdb.set_trace()
                #plt.close()
        print(' -> tau loop took ', time.time() - tau_start)

                    
    # put everything in a dictionary and save it
    readme = ['The model_mags dictionary has keys corresponding to filter(s),',
                  'each has an array with dimensions as follows:',
                  'index 1 (n='+str(len(tau_list))+'): match to tau_list',
                  'index 2 (n='+str(len(av_list))+'): match to av_list',
                  'index 3 (n='+str(len(age_list))+'): match to age_list',
                  'index 4 (n='+str(len(bump_list))+'): match to bump_list',
                  'index 5 (n='+str(len(rv_list))+'): match to rv_list',
                  '',
                  'the model_*_mass array dimensions are: ',
                  'index 1 (n='+str(len(tau_list))+'): match to tau_list',
                  'index 2 (n='+str(len(age_list))+'): match to age_list',
                  '',
                  'The model_mags are absolute AB magnitudes for a galaxy', 
                  'with mass = 1 m_sun.  The model_stellar_mass contains',
                  "the fraction of that mass that's in stars.", 
                  '',
                  'This assumes a metallicity of z=0.'+metallicity+'.',
                  '',
                  'This assumes a distance of 10pc.' ]

    # change tau list to numpy array
    tau_list = np.array(tau_list).astype(np.float)

    model_info = {'tau_list':tau_list, 'age_list':age_list,
                      'av_list': av_list, 'rv_list':rv_list, 'bump_list':bump_list,
                      'model_mags':model_mags, 'model_stellar_mass':model_stellar_mass,
                      'model_remnant_mass':model_remnant_mass, 'model_gas_mass':model_gas_mass,
                      'filter_list':filter_label, 'lambda_list':lambda_list,
                      'readme':readme}
    pickle_file = open('model_grid.pickle','wb')
    pickle.dump(model_info, pickle_file)
    pickle_file.close()


    
                    
def pegase_file(pegase_path, metallicity, sfh):
    """
    Get the file name for a particular PEGASE model

    Parameters
    ----------
    pegase_path : string
       full path to the PEGASE model folders

    metallicity : string
       one of ['005','01','015','02','025','035','050']

    sfh : string
       one of the star formation history strings


    Returns
    -------
    file_path : string
       full path to the PEGASE model file

    """

    # the directories for each metallicity
    model_directory = ['G135z005/','G135z01/','G135z015/','G135z02/','G135z025/','G135z035/','G135z05/']

    # check for valid metallicity
    try:
        match = ['005','01','015','02','025','035','050'].index(metallicity)
        folder = model_directory[match]
    except:
        print("pegase_file: please choose metallicity from ['005','01','015','02','025','035','050']")
        return None

    # check for valid SFH
    try:
        ['b','0110','0140','0155','0170','0185','0200','0215','0235','0250','0275', 
             '0300','0350','0400','0500','0600','0750','1000','1500','3500', 
             'const','g020','g040','g075','g150'].index(sfh)
    except:
        print('pegase_file: please choose valid SFH string')
        return None

    # return the path
    file_path = pegase_path + folder + 'G135z' + metallicity + '.' + sfh + '.dat'
    return file_path
    


    
def writespec_info(spec_file):
    """
    Intended to replicate the functionality of writespec_info.pro,
    which assembles the info I actually want from the giant collection
    in specread

    Parameters
    ----------
    spec_file : string
       full path+name of the PEGASE file

    Returns
    -------
    lcont : array of floats
       wavelengths for continuum

    age : array of floats
       list of ages corresponding to each spectrum

    spec : 2D array of floats
       the spectra, with dimensions of [len(age), len(lcont)]

    m_stellar : array of floats
       list of stellar masses associated with each spectrum

    m_remnant : array of floats
       list of remnant masses associated with each spectrum

    m_gas : array of floats
       list of gas masses associated with each spectrum

    """

    # extract the info from the PEGASE file
    lcont, lline, data = specread(spec_file)

    # number of wavelengths
    ncont = len(lcont)
    nlines = len(lline)

    # age info
    age = np.array( [data[i]['TIME'] for i in range(len(data))] )
    n_age = len(age)

    # assorted masses
    m_stellar = np.array( [data[i]['NMSTAR'] for i in range(len(data))] )
    m_remnant = np.array( [data[i]['NMWD'] for i in range(len(data))] ) + np.array( [data[i]['NMNSBH'] for i in range(len(data))] )
    m_gas = np.array( [data[i]['NMGAS'] for i in range(len(data))] )

    # make a spectral grid
    # this requires adding the lines into the continuum
    spec = np.zeros((n_age, ncont))

    for k in range(n_age):
        
        # grab the continuum spectrum for this age
        spec[k,:] = data[k]['CONT']

        # insert each line
        for i in range(nlines):
            # close continuum wavelength
            match = np.max(np.where( lcont <= lline[i] )[0])
            # wavelength spacing at this spot
            wavelength_spacing = lcont[match+1]-lcont[match]
            # spectral value associated with the line
            line_value = data[k]['LINES'][i]
            # add it to the spectrum
            spec[k,match] = spec[k,match] + line_value/wavelength_spacing
        

    # return stuff
    return lcont, age, spec, m_stellar, m_remnant, m_gas


def specread(spec_file):
    """
    Intended to replicate the functionality of specread.pro, which parses and
    extracts all the numbers from the PEGASE output file.

    Parameters
    ----------
    spec_file : string
       full path+name of the PEGASE file

    Returns
    -------
    lcont : array of floats
       wavelengths for continuum

    lline : array of floats
       wavelengths for lines
    
    data : list of dictionaries
       all of the model info

    """

    # open the file
    f = open(spec_file,'r')

    # skip the informational header (which ends at the ***)
    temp = ''
    while temp != '************************************************************\n':
        temp = f.readline()

    # the next line is the numbers of models and wavelengths
    ntstep, ncont, nlines = [x.strip() for x in f.readline().split(' ') if x != '']
    ntstep = int(ntstep)
    ncont = int(ncont)
    nlines = int(nlines)
    
    # read in the continuum wavelengths
    lcont = spec_chunk(f, ncont)

    # read in the line wavelengths
    llines = spec_chunk(f, nlines)

        
    # now we get to the model grids

    # initialize a list that will hold dictionaries
    data = []

    
    for i in range(ntstep):

        # get the first line of physical parameters
        line1 = [x.strip() for x in f.readline().split(' ') if x != '']
        # get the second line of physical parameters
        line2 = [x.strip() for x in f.readline().split(' ') if x != '']
        
        # read in the continuum spectrum
        cont = spec_chunk(f, ncont)
        # read in the line strengths
        lines = spec_chunk(f, nlines)
        
        # put it all in a dictionary
        data.append({'TIME':float(line1[0]),
                         'NMGAL':float(line1[1]),
                         'NMSTAR':float(line1[2]),
                         'NMWD':float(line1[3]),
                         'NMNSBH':float(line1[4]),
                         'NMSUBS':float(line1[5]),
                         'NMGAS':float(line1[6]),
                         'ZISM':float(line1[7]),
                         'ZSTARMASS':float(line1[8]),
                         'ZSTARLBOL':float(line1[9]),
                         'NLBOL':float(line2[0]),
                         'TAUV':float(line2[1]),
                         'DUSTBOL':float(line2[2]),
                         'NSFR':float(line2[3]),
                         'NLYMAN':float(line2[4]),
                         'NSNII':float(line2[5]),
                         'NSNIA':float(line2[6]),
                         'AGEMASS':float(line2[7]),
                         'AGELBOL':float(line2[8]),
                         'CONT':cont,
                         'LINES':lines } )

    # close the file
    f.close()

    # return things
    return lcont, llines, data


def spec_chunk(the_file, n_points):
    """
    Helper function for specread

    Reads in the appropriate number of lines (each 5 items long) for either the
    list of wavelengths or the spectrum associated with those wavelengths

    This is non-negligible because there may be some extra items that don't fill
    up a full line, but still need to be broken up appropriately

    Parameters
    ----------
    the_file : file object
       the opened file (from specread)
    n_points : integer
       the number of total items that need to be read in

    Returns
    -------
    spec : array of floats
       the numbers (either wavelengths or spectrum)

    """

    # initialize the array
    spec = np.zeros(n_points)
    # - fully populated lines
    for i in range(int(n_points/5)):
        temp = [x.strip() for x in the_file.readline().split(' ') if x != '']
        spec[5*i] = float(temp[0])
        spec[5*i+1] = float(temp[1])
        spec[5*i+2] = float(temp[2])
        spec[5*i+3] = float(temp[3])
        spec[5*i+4] = float(temp[4])
    # - any partial lines
    if n_points % 5 == 1:
        temp = [x.strip() for x in the_file.readline().split(' ') if x != '']
        spec[n_points-1] = float(temp[0])
    if n_points % 5 == 2:
        temp = [x.strip() for x in the_file.readline().split(' ') if x != '']
        spec[n_points-2] = float(temp[0])
        spec[n_points-1] = float(temp[1])
    if n_points % 5 == 3:
        temp = [x.strip() for x in the_file.readline().split(' ') if x != '']
        spec[n_points-3] = float(temp[0])
        spec[n_points-2] = float(temp[1])
        spec[n_points-1] = float(temp[2])
    if n_points % 5 == 4:
        temp = [x.strip() for x in the_file.readline().split(' ') if x != '']
        spec[n_points-4] = float(temp[0])
        spec[n_points-3] = float(temp[1])
        spec[n_points-2] = float(temp[2])
        spec[n_points-1] = float(temp[3])

    return spec







if __name__ == '__main__':

    create_grids()
