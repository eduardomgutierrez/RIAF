import numpy as np
import scipy.constants as const

# CONSTANTS [CGS]

G = const.gravitational_constant * 1.0e3
cLight = const.speed_of_light * 1.0e2
cLight2 = cLight*cLight
solarMass = 1.98847e33
eddAccRate = 1.39e18

def virialTemp(r):
    return 3.6e12/r

adafFile = 'adafFile.txt'
adafParameters = 'adafParameters.txt'

# Fix parameters
coronaFactor = 1.0
SSDdisk = 1
pIndex = 0.1
correctorAccRate = 0.01

# INITIAL PARAMETERS

blackHoleMass = 1.3e8       # [Msol]
eddAccRate = eddAccRate * blackHoleMass
accRateNorm = 1.2e-1          # [MdotEdd]
rOut = 1e3                  # [Rs]
accRateOut = accRateNorm * eddAccRate * coronaFactor
accRateCD = accRateNorm * eddAccRate
innerRadiusSSD = 3.0        # [Rs]

# Adimensional Disk parameters
beta = 0.9
alpha = 0.3
delta = 0.1
s = 0.1

# EIGENVALUES
log10j0 = np.log10(2.2163)
log10j1 = np.log10(2.91)

# Boundary conditions

# rOut ~ 10^2 rS
#temp_i_Out = 0.6*virialTemp(rOut)
#temp_e_Out = 0.08*virialTemp(rOut)
#lamda = 0.5

if np.abs(np.log10(rOut/1.0e2)) < np.abs(np.log10(rOut/1.0e4)):
    temp_i_Out = 0.6*virialTemp(rOut)
    temp_e_Out = 0.08*virialTemp(rOut)
    lamda = 0.5
else:
    temp_i_Out = 0.2*virialTemp(rOut)
    temp_e_Out = 0.19*virialTemp(rOut)
    lamda = 0.2
