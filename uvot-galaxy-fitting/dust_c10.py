import numpy as np


def dust_c10(av, rv, bump, wavelength):
    """
    Calculate the extinction curve following Conroy+10, which is a generalization of Cardelli+89

    A couple parts (NIR/optical and FUV) were copied over from
    http://dust-extinction.readthedocs.io/en/latest/_modules/dust_extinction/dust_extinction.html#CCM89
    since it has the equations in nice readable form

    Parameters
    ----------
    av : float
       value of A_V, in magnitudes (set to 1 for extinction to be A_lambda/A_V)

    rv : float
       value of R_V

    bump : float
       strength of 2175A bump (1 = Milky Way in Cardelli+89)

    wavelength : array of floats
       wavelengths (in Angstroms) at which you want to evaluate the curve

    Returns
    -------
    extinction : array of floats
       total extinction (in magnitudes) at the chosen wavelengths (or, if you
       choose av=1, this will be mathematically equivalent to A_lambda/A_V)

    """

    # get wavenumbers in inverse microns
    x = 1e4 / wavelength

    # number of wavelength
    n_lambda = len(wavelength)

    # set up some arrays to hold components of the function
    ax = np.zeros(n_lambda)
    bx = np.zeros(n_lambda)
    fa = np.zeros(n_lambda)
    fb = np.zeros(n_lambda)
    

    # infrared
    ir_match = (x > 0.3) & (x <= 1.1)

    ax[ir_match] = 0.574 * x[ir_match]**1.61
    bx[ir_match] = -0.527 * x[ir_match]**1.61

    # optical/NIR
    opt_match = (x > 1.1) & (x <= 3.3)

    y = x[opt_match] - 1.82

    ax[opt_match] = np.polyval((.32999, -.7753, .01979, .72085, -.02427,
                                    -.50447, .17699, 1), y) 

    bx[opt_match] = np.polyval((-2.09002, 5.3026, -.62251, -5.38434,
                                    1.07233, 2.28305, 1.41338, 0), y)


    # NUV
    nuv_match = (x > 3.3) & (x <= 5.9)

    fa[nuv_match] = (3.3 / x[nuv_match])**6 * \
        (-0.0370 + 0.0469 * bump - 0.601 * bump / rv + 0.542 / rv)

    ax[nuv_match] = 1.752 - 0.316 * x[nuv_match] \
        - 0.104 * bump / ((x[nuv_match] - 4.67)**2 + 0.341) + fa[nuv_match]

    bx[nuv_match] = -3.09 + 1.825 * x[nuv_match] \
        + 1.206 * bump / ((x[nuv_match] - 4.62)**2 + 0.263)

    # far NUV
    fnuv_match = (x > 5.9) & (x <= 8.0)

    fa[fnuv_match] = -0.0447 * (x[fnuv_match] - 5.9)**2 - 0.00978 * (x[fnuv_match] - 5.9)**3
    fb[fnuv_match] = 0.213 * (x[fnuv_match] - 5.9)**2 + 0.121 * (x[fnuv_match] - 5.9)**3

    ax[fnuv_match] = 1.752 - 0.316 * x[fnuv_match] \
        - 0.104 * bump / ((x[fnuv_match] - 4.67)**2 + 0.341) + fa[fnuv_match]

    bx[fnuv_match] = -3.09 + 1.825 * x[fnuv_match] \
        + 1.206 * bump / ((x[fnuv_match] - 4.62)**2 + 0.263) + fb[fnuv_match]

    # FUV
    fuv_match = (x > 8.0) & (x <= 10.0)

    y = x[fuv_match] - 8.0
    ax[fuv_match] = np.polyval((-.070, .137, -.628, -1.073), y)
    bx[fuv_match] = np.polyval((.374, -.42, 4.257, 13.67), y)
        

    # calculate extinction
    extinction = (ax + bx / rv) * av


    return extinction
