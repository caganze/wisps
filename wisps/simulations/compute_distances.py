
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


pnts=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
#print (pnts[0].survey)
COORDS=SkyCoord([p.coord for p in pnts ])
galc=COORDS.transform_to('galactic')

LBS=np.vstack([[x.coord.galactic.l.radian,x.coord.galactic.b.radian] for x in pnts ])

LS=galc.l.radian
BS=galc.b.radian

#OBSERVED_DIST=np.concatenate(np.array([v for v in pnts[0].dist_limits.values()]))

def density_function(r, z, h=300.):
    
    """
    A custom juric density function that only uses numpy arrays for speed
    All units are in pc
    """
    l = 2600. # radial length scale of exponential thin disk 
    zpart=np.exp(-abs(z-Zsun)/h)
    rpart=np.exp(-(r-Rsun)/l)
    return zpart*rpart


def sample_distances(nsample=1000, h=300, dmin=0, dmax=5000):
    """
    sample the galaxy given a scale height
    
    """
    #add an option for sampling a uniform distribution for scale-heights
    def logp(l, b, r, z, d, h):
        return np.log((d**2)*density_function(r, z, h))

    with pm.Model() as model:
        l=pm.Uniform('l', lower=-2*np.pi, upper=2*np.pi, testval=np.pi/2, observed=LS)
        b=pm.Uniform('b', lower=-2*np.pi, upper=2*np.pi, testval=np.pi/3, observed=BS)
    
        d=pm.Uniform('d', lower=0., upper=6000, testval=500., shape=BS.shape)
        
        x=pm.Deterministic('x',  Rsun-d*np.cos(b)*np.cos(l))
        y=pm.Deterministic('y', -d*np.cos(b)*np.sin(l))
        r=pm.Deterministic('r', (x**2+y**2)**0.5 )
        z=pm.Deterministic('z', Zsun+ d * np.sin(b))

        #add an option for sampling a uniform distribution
        #if h=='uniform':
        #    h=pm.Uniform('h', lower=h_bounds[0], upper=h_bounds[-1])

        like = pm.Potential('likelihood', logp(l, b, r, z, d, h))

        trace = pm.sample(draws=int(nsample), cores=4, tune=int(nsample/20),
            discard_tuned_samples=True, step=pm.Metropolis())

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

def save_all_stuff():
    #sample the galactic structure model
    import pickle
    #some re-arragments because the limiting distance depends on the pointing
    dist_arrays=pd.DataFrame.from_records([x.dist_limits for x in pnts]).applymap(lambda x:np.vstack(x).astype(float))
    DISTANCE_LIMITS={}
    for s in wispsim.SPGRID:
        DISTANCE_LIMITS[s]=dist_arrays[s].mean(axis=0)

    for h in wispsim.HS:
        print (h)
        dis={}
        for s in DISTANCE_LIMITS.keys():
            print (s)
            dlts=np.array(DISTANCE_LIMITS[s]).flatten()
            dx= sample_distances(nsample=1e3, h=h, dmin=dlts[1]/2, dmax=2*dlts[0])
            dis.update({s: dx})
            jbhknj
        DISTANCE_SAMPLES.update({h: dis})

    with open(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', 'wb') as file:
                       pickle.dump(DISTANCE_SAMPLES,file)
if __name__ =='__main__':
    save_all_stuff()
    #import wisps.simulations.effective_numbers as eff
    #eff.simulation_outputs(recompute=True, hs=wispsim.HS)

