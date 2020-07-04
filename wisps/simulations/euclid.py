
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

import pandas as pd

import pymc3 as pm
import seaborn as sns 

import numba
from scipy import integrate

from .binaries import make_systems
from wisps.utils.tools import get_distance


from tqdm import tqdm
import wisps
import wisps.simulations as wispsim

#constant distance

EUCLID_SOUTH=SkyCoord(l=24.6*u.deg, b=-82.0*u.deg , frame='galactic').galactic
EUCLID_NORTH=SkyCoord("18:0:0 66:33:0", obstime="J2000", unit=u.deg).galactic
EUCLID_FORNAX=SkyCoord("3:32:28.0 -27:48:30" , obstime="J2000", unit=u.deg).galactic

#mag limits

EUCLID_MAG_LIMITS={'J': 27., 'H': 27.}
#absol=#wisps.absolute_magnitude_jh(wispsim.SPGRID)[1]
#RELJ=wisps.POLYNOMIAL_RELATIONS['abs_mags']['EUCLID_J']
RELH=wisps.POLYNOMIAL_RELATIONS['abs_mags']['EUCLID_H']

absol=(RELH[0])(np.random.normal(wispsim.SPGRID, RELH[1]))

DMAXS=dict(zip(wispsim.SPGRID, (wisps.get_distance(absol, np.ones_like(absol)*EUCLID_MAG_LIMITS['H']))))


#constants
Rsun=wispsim.Rsun
Zsun=wispsim.Zsun


def distance_sampler(l, b, nsample=1000, h=300, dmax=1000):
    """
    sample the galaxy given a scale height
    l and b must be in radian
    """
    def logp(l, b, r, z, d, h):
        return np.log((d**2)*wispsim.density_function(r, z, h))

    with pm.Model() as model:
 
        d=pm.Uniform('d', lower=0., upper=dmax, testval=10.,)
        
        x=pm.Deterministic('x',  Rsun-d*np.cos(b)*np.cos(l))
        y=pm.Deterministic('y', -d*np.cos(b)*np.sin(l))
        r=pm.Deterministic('r', (x**2+y**2)**0.5 )
        z=pm.Deterministic('z', Zsun+ d * np.sin(b))

        like = pm.DensityDist('likelihood', logp, observed={'l':l, 'b':b,
                             'r': r, 'z': z, 'd':d, 'h':h})

        trace = pm.sample(draws=int(nsample), cores=4, step=pm.Metropolis(), tune=int(nsample/20), discard_tuned_samples=True)

    return trace

@np.vectorize
def euclid_selection_function(j, h):
	#a simple step-function selection function based on mag cuts
	s=0.
	if j <EUCLID_MAG_LIMITS['J']:
		s=1.
	if h<EUCLID_MAG_LIMITS['H']:
		s=1.
	return s



def expected_numbers(model, field='fornax', h=300):
	#compute exepected numbers in euclid fields based on different model based on a mode
    #spectral type
    syst=make_systems(model_name=model, bfraction=0.2)

    sortedindx=np.argsort((syst['system_spts']).flatten())
    spts=((syst['system_spts']).flatten())[sortedindx]
    #


    round_spts=np.round(spts).astype(float).flatten()
    print (round_spts.shape)

    #distances
    dists=None
    ds=np.zeros(len(spts))

    coordinate_field=None
    if field=='fornax':
        coordinate_field=EUCLID_FORNAX

    if field=='south':
        coordinate_field=EUCLID_SOUTH
    
    if field=='north':
        coordinate_field=EUCLID_NORTH
        
    for k in DMAXS.keys():
        trace=distance_sampler(coordinate_field.l.radian, coordinate_field.b.radian, dmax=DMAXS[k], nsample=1000, h=h)
        indx= (round_spts==k)
        ds[indx]=np.random.choice(trace['d'].flatten(), len(round_spts[indx]))


    absjs, abshs=wisps.absolute_magnitude_jh(spts)
 
    dists=ds

    appjs=absjs+5*np.log10(dists/10.0)
    apphs=abshs+5*np.log10(dists/10.0)
  
    #selection probabilities
    s=euclid_selection_function(appjs, apphs)

    #teffs are for normalizing the LF 
    return {'spt': spts, 'ds': dists,  'j':appjs, 'h':apphs, 'prob': s, 'teff': ((syst['system_teff']).flatten())[sortedindx]}