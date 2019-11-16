
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


#imports
#----------------------

#constants
hs =  [100, 250, 275, 300, 325 , 350, 1000] 
Rsun=83000.
Zsun=25.
spgrid=wispsim.SPGRID
#-----------------------

#read-in the pointings
pnts=pd.read_pickle(wisps.OUTPUT_FILES+'/bayesian_observed_pointings.pkl')
COORDS=SkyCoord([p.coord for p in wisps.OBSERVED_POINTINGS ])

galc=COORDS.transform_to('galactic')

LS=galc.l.radian

BS=galc.b.radian

OBSERVED_DIST=np.concatenate(np.array([v for v in pnts[0].dist_limits.values()]))
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

@numba.jit
def custom_volume_correction(coordinate,dmin, dmax, h):
    nsamp=100
    ds = np.linspace(dmin,dmax,nsamp)
    ga=coordinate.transform_to('galactic')
    rd=np.sqrt( (ds * np.cos( ga.b.radian) )**2 + Rsun * (Rsun - 2 * ds * np.cos( ga.b.radian ) * np.cos( ga.l.radian ) ) )
    zd=Zsun+ ds * np.sin( ga.b.radian - np.arctan( Zsun / Rsun) )
    rh0=density_function(rd, zd,h=h )
    num=integrate.trapz(rh0*(ds**2), x=ds)
    den=((dmax-dmin)**3)
    return  abs(num/den)

def computer_volume(pnt):
        """
        given area calculate the volume
        """
        volumes={}
        for k in spgrid:
            vcs=[]
            for h in hs:
                vc=custom_volume_correction(pnt.coord,  pnt.dist_limits[k][1], pnt.dist_limits[k][0], h)
                vcs.append(vc)
            volumes['vc_{}'.format(str(k))]=vcs
            volumes[k]= np.array(vcs)*0.3333333333333*(pnt.dist_limits[k][0]**3-pnt.dist_limits[k][1]**3)

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

    volumes=[computer_volume(pnt) for pnt in pnts]
    dists=np.array(dists)

    dist_dict=dict(zip(hs, dists))
    rs_dict=dict(zip(hs, rs))
    zs_dict=dict(zip(hs, zs))


    full_dict={'volumes': volumes, 'distances': dist_dict, 'rs': rs_dict, 'zs': zs_dict}

    import pickle
    with open(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', 'wb') as file:
               pickle.dump(full_dict,file)

    with open(wisps.OUTPUT_FILES+'/distance_limits.pkl', 'wb') as file:
               pickle.dump(pnts[0].dist_limits,file)

    import wisps.simulations.effective_numbers as eff
    eff.simulation_outputs(recompute=True)


