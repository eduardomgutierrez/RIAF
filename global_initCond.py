import numpy as np
import scipy.constants as const

# CONSTANTS [CGS]

G = const.gravitational_constant * 1.0e3
cLight = const.speed_of_light * 1.0e2
cLight2 = cLight*cLight
solarMass = 1.98847e33

def virialTemp(r):
    return 3.6e12/r

adafFile = 'adafFile.txt'
adafParameters = 'adafParameters.txt'

# INITIAL PARAMETERS

blackHoleMass = 4.0e6   # [Msol]
beta = 0.9
alpha = 0.1
delta = 0.5

rOut = 1.0e4
eddAccRate = 1.39e18 * blackHoleMass
accRateNorm = 1.0e-5
accRateOut = accRateNorm*eddAccRate
s = 0.3

# EIGENVALUES

log10j0 = np.log10(1.4)
log10j1 = np.log10(1.43)

# INITIAL VALUES

# rOut ~ 10^2 rS
#temp_i_Out = 0.6*virialTemp(rOut)
#temp_e_Out = 0.08*virialTemp(rOut)
#lamda = 0.5

# rOut ~ 10^4 rS
temp_i_Out = 0.2*virialTemp(rOut)
temp_e_Out = 0.19*virialTemp(rOut)
lamda = 0.2
