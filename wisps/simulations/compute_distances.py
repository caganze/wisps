
import numpy as np
import matplotlib.pyplot as plt
import wisps
import wisps.simulations as wispsim
import matplotlib as mpl
import astropy.units as u
from astropy.coordinates import SkyCoord
import theano
import theano.tensor as T
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

Rsun=83000.
Zsun=25.
spgrid=wispsim.SPGRID
#-----------------------

#read-in the pointings
pnts=wisps.OBSERVED_POINTINGS
COORDS=SkyCoord([p.coord for p in wisps.OBSERVED_POINTINGS ])

galc=COORDS.transform_to('galactic')

LS=galc.l.radian

BS=galc.b.radian

hs=[200, 250, 275, 300, 325, 350, 1000]
#OBSERVED_DIST=np.concatenate(np.array([v for v in pnts[0].dist_limits.values()]))
#---------------------------

#define functions
#-------------------------------------------
def density_function(r, z, h=300.):
    
    """
    A custom juric density function that only uses numpy arrays for speed
    All units are in pc
    """
    l = 2600. # radial length scale of exponential thin disk 
    
    zpart=(1./np.cosh(abs(z-Zsun)/(2*h)))**2
    rpart=np.exp(-(r-Rsun)/h)
    
    return zpart*rpart

def sample_distances(nsample=1000, h=300):
    """
    sample the galaxy given a scale height
    
    """
    def logp(r, z, d):
        return np.log10((d**2)*density_function(r, z, h))

    with pm.Model() as model:
        
        l=pm.Uniform('l', lower=np.nanmin(LS), upper=np.nanmax(LS) , observed=LS)
        b=pm.Uniform('b', lower=np.nanmin(BS), upper=np.nanmax(BS),  observed=BS)
        d=pm.Uniform('d', lower=h/100, upper=10*h, testval=5*h, shape=len(BS))

        r=pm.Deterministic('r', np.sqrt( (d * np.cos( b ) )**2 + Rsun * (Rsun - 2 * d * np.cos( b ) * np.cos( l ) ) ))
        z=pm.Deterministic('z', Zsun+ d * np.sin( b - np.arctan( Zsun / Rsun) ))
        
        like = pm.DensityDist('likelihood', logp, observed={
                             'r': r, 'z': z, 'd':d})

        trace = pm.sample(draws=int(nsample), cores=2, step=pm.Metropolis())
    return trace



#measure volumes with changing scale heights

def custom_volume(coordinate,dmin, dmax, h):
    nsamp=1000
    ds = np.linspace(dmin,dmax,nsamp)
    rd=np.sqrt( (ds * np.cos( coordinate.galactic.b.value  ) )**2 +
               Rsun * (Rsun - 2 * ds* np.cos( coordinate.galactic.b.value ) * np.cos( coordinate.galactic.l.value ) ) )
    zd=Zsun+ ds * np.sin( coordinate.galactic.b.value - np.arctan( Zsun / Rsun) )
    rh0=density_function(rd, zd,h=h )
    val=integrate.trapz(rh0*(ds**2), x=ds)

    return val

def get_accurate_relations(x, rel, rel_unc):
    #use monte-carlo error propgation
    vals=np.random.normal(rel(x), rel_unc, 100)
    return np.nanmean(vals)


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
        pol=rels['sp_F160W']
        pol_unc=rels['sigma_sp_F160W']
        maglmts= wisps.MAG_LIMITS['wisps']['F160W']
    if pnt.survey=='hst3d':
        pol=rels['sp_F160W']
        pol_unc=rels['sigma_sp_F160W']
        maglmts=wisps.MAG_LIMITS['hst3d']['F160W']

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

def computer_volume(pnt):
        """
        given area calculate the volume
        """
        volumes={}
        dist_limits=compute_distance_limits(pnt)
        for k in spgrid:
            vs=[]
            for h in hs:
                v=custom_volume(pnt.coord,  dist_limits[k][1], dist_limits[k][0], h)*wispsim.SOLID_ANGLE
                vs.append(v)
            volumes[k]= np.array(vs)

    

        return volumes




#----------------------------------
#save stuff 
if __name__ =='__main__':
    
    #sample the galactic structure model
    traces=[]
    for h in hs:
        traces.append(sample_distances(nsample=20000, h=h))


    dists=np.array([t['d'] for t in traces])
    rs=np.array([t['r'] for t in traces])
    zs=np.array([t['z'] for t in traces])

    #--------------------------------------------
    volumes=[computer_volume(pnt) for pnt in tqdm(pnts)]
    dist_limits=compute_distance_limits(pnts[0])
    dists=np.array(dists)

    dist_dict=dict(zip(hs, dists))
    rs_dict=dict(zip(hs, rs))
    zs_dict=dict(zip(hs, zs))


    full_dict={'volumes': volumes, 'distances': dist_dict, 'rs': rs_dict, 'zs': zs_dict}

    import pickle
    with open(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', 'wb') as file:
               pickle.dump(full_dict,file)

    with open(wisps.OUTPUT_FILES+'/distance_limits.pkl', 'wb') as file:
               pickle.dump(dist_limits,file)

    import wisps.simulations.effective_numbers as eff
    eff.simulation_outputs(recompute=True, hs=hs)


