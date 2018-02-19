This is an SED fitter that combines PEGASE.2 spectral models with a grid of attenuation curve parameters (Av, Rv, 2175A bump).  This is particularly useful for UVOT data that can independently constrain Rv and the bump, rather than either assuming an attenuation curve or choosing the best fit of a handful of curves.  It uses MCMC, but there is also a chi^2 fitter for quick-look results.

The PEGASE.2 grids (created a long time ago by Erik Hoversten) can be downloaded here:
<https://stsci.box.com/v/pegase-grids>
They should be placed in ``uvot-galaxy-fitting/uvot-galaxy-fitting/pegase_grids``.

The code and documentation are in active development.  Please consult @lea-hagen before using this.


How to use
----------

Required packages: pysynphot, emcee, corner

Installation: Either download or clone the repository.  You can keep the code wherever you like as long as it's in your python path.

This code does two things: (1) generating a 5-dimensional grid of models varying tau, age, Av, Rv, and 2175A bump strength, and (2) finding the best fit for an SED.  This is intended to be a broad overview, and the codes each have docstrings which will be helpful for details.

In your working directory (again, that can be anywhere), create a file called model_parameters.py (example is included).  It should have NumPy arrays of the grid values for Av, Rv, and 2175A bump strength.  The grid values for tau and age are currently hardcoded.

Prior to running anything, you will need to choose a way to label observations to distinguish between filters.  This would ideally be a filter name, but it could be anything you want, as long as it's unique between filters.  These labels will be attached to both your photometry and the model grid.

To run ``create_grids``, the inputs are a list of these filter labels, a list of files for the corresponding filter transmission curves, and a metallicity.  The code will create a dictionary for which each key is a filter label and each value is a 5-dimensional array of model magnitudes.  It will also save other information (model parameters, average wavelengths, etc).  All of this will be bundled into a single dictionary and saved as a pickle file with the name ``model_grid_Z.pickle``, where Z represents the metallicity.  If you later wish to add another filter to an existing grid, it will append it, rather than regenerating the whole grid.

Next, do the modeling.  The inputs are the AB magnitudes, errors, metallicity, and a label.  The photometry inputs are formatted as dictionaries in which the key is the filter label and the value is the AB magnitude or error.  Currently, only one data point per filter is supported.  The filter label can be any that exists in the grid for the given metallicity.  The label can be any string (likely a galaxy name) that will be used for the output file names.  For the chi2 quick look fitter, there will be 4 outputs: a pickle file with the chi2 grid and values (you shouldn't need to look at this), a text file with the best-fit values, a PDF with a plot of the input SED and best-fit SED, and a PDF with 2D projections of the 5D chi2 space.  Each of these will be organized into folders that the code will create.
