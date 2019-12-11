

################################
# population simulations routines

##############################

#imports
#import numpy as np
import splat.simulate as spsim
import splat.photometry as sphot
import splat.core as spl
import splat.empirical as spem
import splat.simulate as spsim
import splat.evolve as spev
import bisect
import astropy.units as u
import scipy
from .initialize import *
#import wisps
#from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numba 
import emcee
#from emcee import PTSampler
#import warnings; warnings.simplefilter('ignore')
from astropy.coordinates import SkyCoord
import scipy.integrate as integrate
import random
import pymc3 as pm

from ..utils.tools import get_distance




from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial


class Pointing(object):
    ## a pointing object making it easier to draw samples
    
    def __init__(self, **kwargs):
        #only input is the direction
        self.coord=kwargs.get('coord', None)
        self.survey=kwargs.get('survey', None)
        self.name=kwargs.get('name', None)

def make_pointings():

    maglimits=wisps.MAG_LIMITS
    
    obs=pd.read_csv(wisps.OUTPUT_FILES+'//observation_log_with_limit.csv')

    def make_pointing(ra, dec, survey, name):
        coord=SkyCoord(ra=ra*u.deg,dec=dec*u.deg )
        return Pointing(coord=coord, survey=survey, name=name)

    def get_survey(pointing):
        if pointing.startswith('par'):
            return 'wisps'
        else:
            return 'hst3d'

    ras=obs['ra (deg)']
    decs=obs['dec(deg)']
    surveys=obs.pointing.apply(get_survey)

    pnts=[make_pointing(ra, dec, survey, name) for ra, dec, survey, name in zip(ras, decs, surveys, obs.pointing.values)]

    import pickle

    output_file=wisps.OUTPUT_FILES+'/pointings.pkl'
    with open(output_file, 'wb') as file:
        pickle.dump(pnts,file)
        
def splat_teff_to_spt(teff):
    rel=splat.SPT_TEFF_RELATIONS['pecaut']
    spt_sorted_idx=np.argsort(rel['values'])
    return np.interp(teff, np.array(rel['values'])[spt_sorted_idx], np.array(rel['spt'])[spt_sorted_idx])
       
#simulate spectral types
def simulate_spts(**kwargs):
    """
    simulate a distribution of spectral types using a mass function and evolutionary models
    """
    recompute=kwargs.get('recompute', False)
    
    if recompute:

        norm_range = [0.01, 0.075]
        norm_density = 0.1
        nsim = kwargs.get('nsample', 1e5)
        spts=np.arange(17, 42)

        # simulation
        masses = spsim.simulateMasses(nsim,range=[0.001,0.15],distribution='power-law',alpha=0.5)
        norm = norm_density/len(masses[np.where(np.logical_and(masses>=norm_range[0],masses<norm_range[1]))])


        ages=[]
        teffs=[]
        spts=[]

        #uniform distribution
        ages_unif= spsim.simulateAges(nsim,range=[0.1,10.], distribution='uniform')
        teffs_unif = spev.modelParameters(mass=masses,age=ages_unif, set='baraffe03')['temperature'].value
        spts_unif = splat_teff_to_spt(teffs_unif)

   
        #rujopakarn
        ages_ruj= spsim.simulateAges(nsim,range=[0.1,10.], distribution='rujopakarn')
        teffs_ruj = spev.modelParameters(mass=masses,age=ages_ruj, set='baraffe03')['temperature'].value
        spts_ruj = splat_teff_to_spt(teffs_ruj)

        #aumer
        ages_aum= spsim.simulateAges(nsim,range=[0.1,10.], distribution='aumer')
        teffs_aum = spev.modelParameters(mass=masses,age=ages_aum, set='baraffe03')['temperature'].value
        spts_aum = splat_teff_to_spt(teffs_aum)




        ages.append(ages_unif)
        spts.append(spts_unif)
        teffs.append(teffs_unif)

        ages.append(ages_ruj)
        spts.append(spts_ruj)
        teffs.append(teffs_ruj)

        ages.append(ages_aum)
        spts.append(spts_aum)
        teffs.append(teffs_aum)

        
        values={'mass': masses, 'ages':np.array(ages), 'teffs':np.array(teffs), 'spts':np.array(spts), 'norm':norm}

        import pickle
        with open(wisps.OUTPUT_FILES+'/mass_age_spcts_2nd.pkl', 'wb') as file:
           pickle.dump(values,file)
    else:
        values=pd.read_pickle(wisps.OUTPUT_FILES+'/mass_age_spcts_2nd.pkl')


    return values