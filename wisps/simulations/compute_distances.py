
import numpy as np
import matplotlib.pyplot as plt
import wisps
import wisps.simulations as wispsim
import matplotlib as mpl
import astropy.units as u
from astropy.coordinates import SkyCoord
import theano
import theano.tensor as tt
import pandas as pd
import pymc3 as pm
import seaborn as sns 
from matplotlib.colors import Normalize
import numba
from scipy import integrate

from wisps.utils.tools import get_distance
from tqdm import tqdm


#imports
#----------------------

#constants
Rsun=wispsim.Rsun
Zsun=wispsim.Zsun

spgrid=wispsim.SPGRID
#-----------------------

#read-in the pointings
pnts=wisps.OBSERVED_POINTINGS
print (pnts[0].survey)
COORDS=SkyCoord([p.coord for p in wisps.OBSERVED_POINTINGS ])
LBS=np.vstack([[x.coord.galactic.l.radian,x.coord.galactic.b.radian] for x in pnts ])


galc=COORDS.transform_to('galactic')

LS=galc.l.radian

BS=galc.b.radian


#OBSERVED_DIST=np.concatenate(np.array([v for v in pnts[0].dist_limits.values()]))
#---------------------------

#define functions

def sample_distances(nsample=1000, h=300):
    """
    sample the galaxy given a scale height
    
    """
    def logprior(l, b):
        return tt.switch(( abs(b) < 0.35),-np.inf, 0)

    def logp(l, b, r, z, d, h):
        return np.log((d**2)*wispsim.density_function(r, z, h))+logprior(l, b)

    with pm.Model() as model:
        l=pm.Uniform('l', lower=-np.pi, upper=np.pi, testval=np.pi/2)
        b=pm.Uniform('b', lower=-np.pi/2, upper=np.pi/2, testval=np.pi/3)
    
        d=pm.Uniform('d', lower=0., upper=5000., testval=500.)
        
        x=pm.Deterministic('x',  Rsun-d*np.cos(b)*np.cos(l))
        y=pm.Deterministic('y', -d*np.cos(b)*np.sin(l))
        r=pm.Deterministic('r', (x**2+y**2)**0.5 )
        z=pm.Deterministic('z', Zsun+ d * np.sin(b))
        
        like = pm.DensityDist('likelihood', logp, observed={'l':l, 'b':b,
                             'r': r, 'z': z, 'd':d, 'h':h})
        trace = pm.sample(draws=int(nsample), cores=2, step=pm.Metropolis())
    return trace



#measure volumes with changing scale heights
#need to change this to directly measuring l and b and 

def compute_distance_limits(pnt):
    """
    computes distance limits based on limiting mags
    """
    rels=wisps.POLYNOMIAL_RELATIONS

    dists=None
    
    #use F140W for 3d-hst pointing and f110w for wisps
    pol=None
    maglmts=None
    pol_unc=None

    if pnt.survey=='wisps':
        pol=rels['sp_F140W']
        pol_unc=rels['sigma_sp_F140W']
        maglmts= wisps.MAG_LIMITS['wisps']['F140W']
    if pnt.survey=='hst3d':
        pol=rels['sp_F140W']
        pol_unc=rels['sigma_sp_F140W']
        maglmts=wisps.MAG_LIMITS['hst3d']['F140W']

    #compute asbmags using abolute mag relations
    absmags=[get_accurate_relations(x, pol, pol_unc) for x in spgrid]
    relfaintmags=np.array([maglmts[0] for s in wispsim.SPGRID])
    relbrightmags=np.array([maglmts[1] for s in wispsim.SPGRID])
    
    #compute distances
    dmins=get_distance(absmags, relbrightmags)
    dmaxs=get_distance(absmags, relfaintmags)

    distances=np.array([dmaxs, dmins]).T

    return dict(zip(wispsim.SPGRID, distances))
            #create a dictionary

#----------------------------------
#save stuff 
if __name__ =='__main__':
    
    #sample the galactic structure model
    traces=[]
    for h in wispsim.HS:
        traces.append(sample_distances(nsample=10000, h=h))


    dists=np.array([t['d'] for t in traces])
    ls=np.array([t['l'] for t in traces])
    bs=np.array([t['b'] for t in traces])

   
    dists=np.array(dists)

    dist_dict=dict(zip(wispsim.HS, dists))
    ls_dict=dict(zip(wispsim.HS, ls))
    bs_dict=dict(zip(wispsim.HS, bs))


    full_dict={ 'distances': dist_dict, 'ls': ls_dict, 'bs': bs_dict}

    import pickle
    with open(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', 'wb') as file:
               pickle.dump(full_dict,file)

    import wisps.simulations.effective_numbers as eff
    eff.simulation_outputs(recompute=True, hs=wispsim.HS)


