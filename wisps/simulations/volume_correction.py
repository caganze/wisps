
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

# imports: splat
from splat.initialize import *
from splat.utilities import *
import splat.empirical as spem
from splat.plot import plotMap
from splat.evolve import modelParameters


#####################################
#                                   #
# Population Simulation routines    #
#                                   #
#####################################


def galactic_density_juric(rc,zc,rho0 = 1./(u.pc**3),report='total',center='sun',unit=u.pc,**kwargs):
    '''
    :Purpose: 
        Returns the local galactic star density at galactic radial (r) and vertical (z) coordinates relative to an assumed "local" density. 
        for the Galaxy model of `Juric et al. (2008, ApJ, 673, 864) <http://adsabs.harvard.edu/abs/2008ApJ...673..864J>`_
        Coordinates are sun-centered unless otherwise specified
    :Required Inputs:
        :param rc: single or array of floating points of galactic radial coordinates, assumed to be in units of pc
        :param zc: single or array of floating points of galactic vertical coordinates, assumed to be in units of pc
    :Optional Inputs:
        :param: rho0 = 1./pc^3: local number density
        :param: center = 'sun': assumed center point, by default 'sun' but could also be 'galaxy'
        :param: report = 'total: what density to report:
            * 'total': (default) report the total galactic number density
            * 'disk' or 'thin disk': report only the thin disk component
            * 'thick disk': report the thick disk component
            * 'halo': report the halo component
            * 'each': return three arrays reporting the thin disk, thick disk, and halo components respectively
        :param: unit = astropy.units.pc: preferred unit for positional arguments
    :Output: 
        Array(s) reporting the number density at the (r,z) coordinates provided in the same units as rho0
    :Example:
        >>> import splat
        >>> import splat.simulate as spsim
        >>> import astropy.units as u
        >>> import numpy
        >>> c = splat.properCoordinates('J05591914-1404488',distance=10.2)
        >>> x,y,z = splat.xyz(c)
        >>> spsim.galactic_density_juric((x**2+y**2)**0.5,z,rho0=1.*(u.pc**(-3)),report='each')
            (<Quantity 0.8232035246365755 1 / pc3>, <Quantity 0.10381465877236985 1 / pc3>, <Quantity 0.004517719384500654 1 / pc3>)
        >>> z = numpy.linspace(0,10,10)
        >>> spsim.galactic_density_juric(z*0,z,unit=u.kpc)
            array([  9.26012756e-01,   5.45786748e-02,   1.28473366e-02,
                     5.34605961e-03,   2.82616132e-03,   1.75923983e-03,
                     1.21099173e-03,   8.82969121e-04,   6.66649153e-04,
                     5.15618875e-04])    
    '''    
# constants
    r0 = (8000.*u.pc).to(unit).value # radial offset from galactic center to Sun
    z0 = (25.*u.pc).to(unit).value  # vertical offset from galactic plane to Sun
    l1 = (2600.*u.pc).to(unit).value # radial length scale of exponential thin disk 
    h1 = (300.*u.pc).to(unit).value # vertical length scale of exponential thin disk 
    ftd = 0.12 # relative number of thick disk to thin disk star counts
    l2 = (3600.*u.pc).to(unit).value # radial length scale of exponential thin disk 
    h2 = (900.*u.pc).to(unit).value # vertical length scale of exponential thin disk 
    fh = 0.0051 # relative number of halo to thin disk star counts
    qh = 0.64 # halo axial ratio
    nh = 2.77 # halo power law index

# note: Juric defines R,Z = R0,0 to be the location of the sun

# check inputs including unit conversion
    if not isinstance(rc,list):
        try: r = list(rc)
        except: r = rc
    else: r = rc
    if not isinstance(r,list): r = [r]
    if isUnit(r[0]): r = [float(d.to(unit).value) for d in r]
    r = numpy.array(r)

    if not isinstance(zc,list):
        try: z = list(zc)
        except: z = zc
    else: z = zc
    if not isinstance(z,list): z = [z]
    if isUnit(z[0]): z = [float(d.to(unit).value) for d in z]
    z = numpy.array(z)

# centering offsets
    if center.lower() == 'sun': 
        r = r+r0
        z = z+z0
#    elif center.lower() == 'galaxy' or center.lower() == 'galactic':
#        z = z-z0


# compute disk fraction
    rhod0 = rho0/(1.+ftd+fh)

# compute number densities of different components
    rhod = rhod0*numpy.exp(-1.*(r-r0)/l1)*numpy.exp(-1.*numpy.absolute(z)/h1)
    rhotd = ftd*rhod0*numpy.exp(-1.*(r-r0)/l2)*numpy.exp(-1.*numpy.absolute(z)/h2)
    rhoh = fh*rhod0*(((r0/(r**2+(z/qh)**2)**0.5))**nh)

# compensate for fact that we measure local density at the sun's position
    if center.lower() == 'sun': 
        rhod = rhod*numpy.exp(z0/h1)
        rhotd = rhotd*numpy.exp(z0/h2)

    if len(r) == 1:
        rhod = rhod[0]
        rhotd = rhotd[0]
        rhoh = rhoh[0]

    rho = rhod+rhotd+rhoh

    if report=='halo': return rhoh
    elif report=='disk' or report=='thin disk': return rhod
    elif report=='thick disk': return rhotd
    elif report=='each': return rhod,rhotd,rhoh
    else: return rho


def volumeCorrection(coordinate,dmax,dmin=0.,model='juric',center='sun',nsamp=1000,unit=u.pc):
    '''
    :Purpose: 
        Computes the effective volume sampled in a given direction to an outer distance value based on an underly stellar density model. 
        This program computes the value of the ratio:
        $\int_0^{x_{max}}{rho(x)x^2dx} / \int_0^{x_{max}}{rho(0)x^2dx}$
    :Required Inputs:
        :param coordinate: a variable that can be converted to an astropy SkyCoord value with `splat.properCoordinates()`_
        :param dmax: the maximum distance to compute to, assumed in units of parsec
    :Optional Inputs:
        :param: model = 'juric': the galactic number density model; currently available:
            * 'juric': (default) `Juric et al. (2008, ApJ, 673, 864) <http://adsabs.harvard.edu/abs/2008ApJ...673..864J>`_ called by `splat.simulate.galactic_density_juric()`_
        :param: center = 'sun': assumed center point, by default 'sun' but could also be 'galaxy'
        :param: nsamp = number of samples for sampling line of sight
        :param: unit = astropy.units.pc: preferred unit for positional arguments
    :Output: 
        Estimate of the correction factor for the effective volume
    :Example:
        >>> import splat
        >>> import splat.simulate as spsim
        >>> c = splat.properCoordinates('J05591914-1404488')
        >>> spsim.volumeCorrection(c,10.)
            1.0044083458899131 # note: slightly larger than 1 because we are going toward Galactic disk
        >>> spsim.volumeCorrection(c,10000.)
            0.0060593740293862081
    .. _`modelParameters()` : api.html#splat.evolve.modelParameters
    .. _`splat.properCoordinates()` : api.html#splat.utilities.properCoordinates
    .. _`splat.simulate.galactic_density_juric()` : api.html#splat.simulate.galactic_density_juric
    '''    
# check inputs
    if not isUnit(unit): unit = u.pc

    try:
        c = splat.properCoordinates(coordinate)
    except: 
        raise ValueError('Input variable {} is not a proper coordinate or list of coordinates'.format(coordinate))
    try:
        x = len(c)
    except:
        c = [c]

    dmx = copy.deepcopy(dmax)
    if isUnit(dmx): dmx = dmx.to(unit).value
    if not isinstance(dmx,float): 
        try: dmx = float(dmx)
        except: raise ValueError('{} is not a proper distance value'.format(dmax))
    if dmx == 0.: return 1.

    dmn = copy.deepcopy(dmin)
    if isUnit(dmn): dmn = dmn.to(unit).value
    if not isinstance(dmn,float): 
        try: dmn = float(dmn)
        except: raise ValueError('{} is not a proper distance value'.format(dmin))

# galactic number density function
    if model.lower() == 'juric':
        rho_function = galactic_density_juric
    elif model.lower() == 'uniform':
        return 1.
    else:
        raise ValueError('\nDo not have galatic model {} for volumeCorrection'.format(model))

# generate R,z vectors
# single sight line & distance
    d = numpy.linspace(dmn,dmx,nsamp)
    rho = []
    for crd in c:
        x,y,z = splat.xyz(crd,distance=d,center=center,unit=unit)
        r = (x**2+y**2)**0.5
        rho.append(rho_function(r,z,rho0=1.,center=center,unit=unit))

    if len(rho) == 1:
        return float(integrate.trapz(rho[0]*(d**2),x=d)/integrate.trapz(d**2,x=d))
    else:
        return [float(integrate.trapz(r*(d**2),x=d)/integrate.trapz(d**2,x=d)) for r in rho]



