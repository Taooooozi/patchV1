import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from grating import *
from ext_signal import *
from scipy.stats import qmc
import sys


def acuity(ecc):
    k = 0.2049795945022049
    log_cpd0 = 3.6741080244555278
    cpd = np.exp(-k*ecc + log_cpd0)
    return 2/cpd/4

def zeroCross(k1, k2, r1, r2):
    return np.sqrt(np.log(k1/k2)/(r2*r2 - r1*r1))*r1*r2


# if __name__ == "main":
if len(sys.argv) < 2:
    print(sys.argv)
    raise Exception('not enough argument for ext_input_debug.py')
else:
    path = sys.argv[1]
    print(path)

K_onC = 29.13289963
K_onS = 23.17378917
K_offC = 22.71914498
K_offS = 12.53276353
r_C = (acuity(0.095) + acuity(0))/2
r_S = r_C*3
cov = 0.53753
SF = 1.0/(zeroCross(K_offC, K_onS*cov, r_C, r_S)*2 + zeroCross(K_onC, K_offS*cov, r_C, r_S)*2)
inputLMS = True
range_deg = 0.0344#0.12672379774682216 # eccentricity from the origin
buffer_deg = acuity(range_deg)*4
SF = 20
TF = 8
time = np.array([1/TF])
neye = 1
center = np.pi/2
wing = np.pi/2
sharpness = 1

n = 1
c = 0.3
P = np.arange(0,1,1)
SF = np.arange(20,30,10)
D = np.arange(0,1,1)
P = P.astype('float32')
SF = SF.astype('float32')
D = D.astype('float32')
crest = np.array([0.5+c, 0.5-c, 0.5])#LMS
valley = np.array([0.5-c, 0.5+c, 0.5])
for sf in SF:
    for orient in D:
        for phase in P:
            video_fn = path + f'static_color_{n}'
            cfg_fn = path + f'static_color_{n}_cfg'
            f = open(cfg_fn+ '.bin', 'wb') # sf,d,p,c
            np.array([sf]).astype('f4').tofile(f)
            np.array([orient*180/np.pi]).astype('f4').tofile(f)
            np.array([phase*180/np.pi]).astype('f4').tofile(f)
            np.array([c]).astype('f4').tofile(f)
            crest.astype('f4').tofile(f)
            valley.astype('f4').tofile(f)
            f.close()
            generate_grating(1.0, sf, TF, orient,32, crest, valley, video_fn, time, phase, sharpness, frameRate =1, ecc = range_deg, buffer_ecc = buffer_deg, gtype='drifting', neye = neye, bar = False, center = center, wing = wing, mask = None, inputLMS = inputLMS,genMovie = True)
            n = n+1
