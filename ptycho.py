#!/usr/bin/env python
# coding: utf-8

# In[1]:

import importlib
import logging
import time

import matplotlib.pyplot as plt
import numpy as np

import tike
import tike.ptycho
import tike.view

# In[2]:

for module in [tike, np]:
    print("{} is version {}".format(module.__name__, module.__version__))

# # Ptychography Reconstruction
#
# This notebook demonstrates a simulated ptychographic reconstruction using tike.
#
# ## Create test data
#
# Make a test data for pytchography reconstruction at one projection angle.

# ### Define the object

# In[3]:

amplitude = plt.imread("../../../tests/data/Cryptomeria_japonica-0128.png")
phase = plt.imread("../../../tests/data/Bombus_terrestris-0128.png") * np.pi
np.min(phase), np.max(phase)

# In[4]:

ntheta = 1  # number angular views
original = np.tile(amplitude * np.exp(1j * phase),
                   (ntheta, 1, 1)).astype('complex64')
tike.view.plot_phase(original[0])
original.shape

# ### Define the probe
#
# Note that the shape of the probe includes many dimensions. These dimensions are for providing unique probes in various situations such as for each projection, for each frame, for each fly scan position, for each incoherent mode.

# In[5]:
pw = 15  # probe width
weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
probe = weights * np.exp(1j * weights * np.pi / 2) * 10
probe = np.tile(probe,
                (ntheta, 1, 1)).astype('complex64')[:, np.newaxis, np.newaxis,
                                                    np.newaxis]
tike.view.plot_phase(probe[0, 0, 0, 0])
probe.shape

# ### Define the trajectory
#
# Each projection may have a different trajectory, but the number of scan positions must be the same. The probe positions that overlap the edge of psi are skipped.

# In[6]:
v, h = np.meshgrid(np.linspace(0,
                               amplitude.shape[0] - pw - 1,
                               21,
                               endpoint=True),
                   np.linspace(0,
                               amplitude.shape[0] - pw - 1,
                               21,
                               endpoint=True),
                   indexing='ij')

# In[7]:
scan = np.tile(np.stack((np.ravel(v), np.ravel(h)), axis=1),
               (ntheta, 1, 1)).astype('float32')
scan.shape

# ## Simulate data acquisition

# In[8]:

# Then what we see at the detector is the wave propagation
# of the near field wavefront
data = tike.ptycho.simulate(detector_shape=pw * 2,
                            probe=probe,
                            scan=scan,
                            psi=original)
np.random.seed(0)
data = np.random.poisson(data)
data.shape

# In[9]:

plt.imshow(np.fft.fftshift(np.log(data[0, 11])))
plt.colorbar()
plt.close('all')
np.min(data), np.max(data)

# ## Reconstruct
#
# Now we need to try and reconstruct psi.

# In[20]:

# Provide initial guesses for parameters that are updated
np.random.seed(4)
init_psi = (np.random.rand(*original.shape) +
            1j * np.random.rand(*original.shape)).astype('complex64')
init_probe = (np.random.rand(*probe.shape) +
              1j * np.random.rand(*probe.shape)).astype('complex64')
np.save(f'data', data)

# In[ ]:

logging.basicConfig(level=logging.INFO)
for algorithm in ['divided', 'combined', 'admm', 'admm1']:
    
    result = {
    'psi': init_psi.copy(),
    'probe': init_probe.copy(),
    'scan': scan,
    'λ': 0,  # parameter for ADMM
    'μ': 0,  # parameter for ADMM
    }

    np.savez(f'{algorithm}.{0:03d}', **result)
    i0 = 0
    for i in np.unique(np.logspace(0, 8, base=2).astype('int')):
        start = time.time()
        result = tike.ptycho.reconstruct(
            data=data,
            **result,
            algorithm=algorithm,
            num_iter=(i - i0),
            recover_probe=True,
            recover_psi=True,
            cg_iter=10,
        )
        stop = time.time()
        i0 = i
        result['time'] = stop - start
        np.savez(f'{algorithm}.{i:03d}', **result)
