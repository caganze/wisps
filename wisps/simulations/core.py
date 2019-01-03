
###modeified volume correction code from splat


# imports: internal
import copy
import glob
import os
import requests
import time

# imports: external
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.constants as constants
from astropy.cosmology import Planck15, z_at_value
from astropy.io import ascii
import pandas
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy
from scipy.interpolate import griddata, interp1d
import scipy.integrate as integrate
import scipy.stats as stats
from tqdm import tqdm

# imports: splat
from splat.initialize import *
from splat.utilities import *
import splat.empirical as spem
from splat.plot import plotMap
from splat.evolve import modelParameters

import splat.simulate as spsim
import splat.evolve as spev
import splat.photometry as sphot

import numpy as np
import pandas as pd

from .initialize import OUTPUT_FILES, CONSTANTS, LUMINOSITY_FUCTION


from wisps.utils import Memoize, profile, memoize_func


#####################################
#                                   #
# Population Simulation routines    #
#                                   #
#####################################

def galactic_density(rc,zc, report='total',center='sun',unit=u.pc,**kwargs):
    #compute juric galactic density function
    r0 = (8000.*u.pc)#.to(unit).value # radial offset from galactic center to Sun
    z0 = (25.*u.pc)#.to(unit).value  # vertical offset from galactic plane to Sun
    l1 = (2600.*u.pc)#.to(unit).value # radial length scale of exponential thin disk 
    h1 = (300.*u.pc)#.to(unit).value # vertical length scale of exponential thin disk 
    ftd = 0.12 # relative number of thick disk to thin disk star counts
    l2 = (3600.*u.pc)#.to(unit).value # radial length scale of exponential thin disk 
    h2 = (900.*u.pc)#.to(unit).value # vertical length scale of exponential thin disk 
    fh = 0.0051 # relative number of halo to thin disk star counts
    qh = 0.64 # halo axial ratio
    p = 2.77 # halo power law index
    rho0 = 1.0/(u.pc**3)

    thin_diskdens=np.exp(-(rc-r0)/l1)*np.exp(-abs(zc-z0)/h1)
    thick_diskdens=ftd*np.exp(-(rc-r0)/l2)*np.exp(-abs(zc-z0)/h2)
    halodens=fh*(r0/(np.sqrt(rc**2+(zc/qh)**2)))**(-p)

    return  thin_diskdens+ thick_diskdens+ halodens

def number(crd, dmin, dmax, weight_function=lambda x: 1.):
    #integrate the number desnity in a particular direction
    d = np.linspace(dmin,dmax,1000)
    rho = []
    x,y,z = splat.xyz(crd,distance=d,center='sun',unit=u.pc)
    r = (x**2+y**2)**0.5
    rho.append(weight_function(d)*galactic_density(r,z))

    if len(rho) == 1:
        return float(integrate.trapz(rho[0]*(d**2),x=d))
    else:
        return [float(integrate.trapz(r*(d**2),x=d)) for r in rho]

def interpolate_luminosity(spt_grid):
    #downsamples? the the luminosity onto a grid
    lf=LUMINOSITY_FUCTION
    f = interp1d(lf.spts.values, lf.lsfim.values)
    return f(spt_grid)


def make_luminosity_function():
    #generate splat luminosity function
    norm_range = [0.09,0.1]
    norm_density = 1.0
    nsim = 1e4
    spts=np.arange(17, 40)

    # simulation
    masses = spsim.simulateMasses(nsim,range=[0.02,0.15],distribution='power-law',alpha=0.5)
    ages = spsim.simulateAges(nsim,range=[0.1,10.],distribution='uniform')
    teffs = spev.modelParameters(mass=masses,age=ages,set='baraffe03')['temperature'].value
    spts = np.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True)[0] for x in teffs])
    norm = norm_density/len(masses[np.where(np.logical_and(masses>=norm_range[0],masses<norm_range[1]))])

    lfsim = []
    spts = spts[np.isfinite(spts) == True]
    for x in tqdm(spts): lfsim.append(len(spts[np.where(np.logical_and(spts>=x,spts<x+1.))]))
    lfsim = np.array(lfsim)*norm

    df=pd.DataFrame()
    df['spts']=spts
    df['lsfim']=lfsim

    df.to_pickle(OUTPUT_FILES+'/luminosity_function.pkl')



