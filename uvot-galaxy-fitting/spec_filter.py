import numpy as np

def spec_filter(spec_lambda, spec_flambda, filter_lambda, filter_trans):
    """
    Get the value for a spectrum observed through a filter

    Note that this assumes interpolating the filter curve at the wavelengths
    in the spectrum, which speeds up computations by ~30% compared to making
    a new lambda_list with both sets of wavelengths

    Parameters
    ----------
    spec_lambda : array of floats
       wavelength of the spectrum (Angstroms)

    spec_flambda : array of floats
       spectrum in erg/s/cm^2/A

    filter_lambda : array of floats
       wavelength of the filter transmission curve (Angstroms)

    filter_trans : array of floats
       transmission of the filter

    Returns
    -------
    output_fnu : float
       the value (in erg/s/cm^2/Hz) of the spectrum through the filter

    """

    # cut off the parts of the spectrum where there's no filter curve info
    keep = (spec_lambda >= np.min(filter_lambda)) & (spec_lambda <= np.max(filter_lambda))
    spec_lambda = spec_lambda[keep]
    spec_flambda = spec_flambda[keep]

    # combine the spectrum and filter wavelengths into one list
    # also make sure there aren't duplicates and it's sorted
    #lambda_list = np.unique(np.append(spec_lambda, filter_lambda))
    lambda_list = spec_lambda

    # interpolate the spectrum/filter at those wavelengths
    #interp_spec_flux = np.interp(lambda_list, spec_lambda, spec_flambda)
    interp_filt_trans = np.interp(lambda_list, filter_lambda, filter_trans)

    # do the maths
    c = 2.998e18 # Ang/sec
    output_fnu = 1/c \
        * np.trapz(spec_flambda * interp_filt_trans * lambda_list, lambda_list) \
        / np.trapz(interp_filt_trans / lambda_list, lambda_list) 
    
    return output_fnu
