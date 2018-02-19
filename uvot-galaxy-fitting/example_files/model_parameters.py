import numpy as np

# list of dust parameters (A_V, R_V, bump strength) you want for your grid
# (at this point, tau/age/metallicity are hard coded)
# - av: 0 to 1 in steps of 0.1, 1 to 4.5 in steps of 0.25
av_list = np.append( np.linspace(0, 1, num=11), np.linspace(1.25, 4.5, num=14) )
# - rv: 1.5 to 5.5 in steps of 0.5
rv_list = np.linspace(1.5, 5.5, num=9)
# - bump: 0 to 2 in steps of 0.2
bump_list = np.linspace(0, 2, num=11)

