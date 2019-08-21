

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

from ..utils.tools import get_distance


sf=pd.read_pickle(wisps.OUTPUT_FILES+'/selection_function.pkl') #my selection function

@numba.jit
def convert_to_rz(ra, dec, dist):
    """
    returns r and z given a distance
    """
    newcoord=SkyCoord(ra=ra, dec=dec, distance=dist*u.pc)
    r=(newcoord.cartesian.x**2+newcoord.cartesian.y**2)**0.5
    z=newcoord.cartesian.z
    return r.to(u.pc).value, z.to(u.pc).value


@numba.vectorize
def l_dwarf_density_function(r, z):
    
    """
    A l dwarfs density function beacsue numba doesn't allow vectorization with kwargs
    """
    ##constants
    r0 = 8000 # radial offset from galactic center to Sun
    z0 = 25.  # vertical offset from galactic plane to Sun
    l1 = 2600. # radial length scale of exponential thin disk 
    h1 = 280.# vertical length scale of exponential thin disk 
    ftd = 0.12 # relative number of thick disk to thin disk star counts
    l2 = 3600. # radial length scale of exponential thin disk 
    h2 = 900. # vertical length scale of exponential thin disk 
    fh = 0.0051 # relative number of halo to thin disk star counts
    qh = 0.64 # halo axial ratio
    nh = 2.77 # halo power law index
    
    dens0=1.0
    
    thindens=dens0*np.exp(-abs(r-r0)/l1)*np.exp(-abs(z-z0)/h1)
    thickdens=dens0*np.exp(-abs(r-r0)/l2)*np.exp(-abs(z-z0)/h2)
    halodens= dens0*(((r0/(r**2+(z/qh)**2)**0.5))**nh)
    
    return thindens+ftd*thickdens+fh*halodens


@numba.vectorize
def juric_density_function(r, z):
    
    """
    A custom juric density function that only uses numpy arrays for speed
    All units are in pc
    """
    ##constants
    r0 = 8000 # radial offset from galactic center to Sun
    z0 = 25.  # vertical offset from galactic plane to Sun
    l1 = 2600. # radial length scale of exponential thin disk 
    h1 = 300.# vertical length scale of exponential thin disk 
    ftd = 0.12 # relative number of thick disk to thin disk star counts
    l2 = 3600. # radial length scale of exponential thin disk 
    h2 = 900. # vertical length scale of exponential thin disk 
    fh = 0.0051 # relative number of halo to thin disk star counts
    qh = 0.64 # halo axial ratio
    nh = 2.77 # halo power law index
    
    dens0=1.0
    
    thindens=dens0*np.exp(-abs(r-r0)/l1)*np.exp(-abs(z-z0)/h1)
    thickdens=dens0*np.exp(-abs(r-r0)/l2)*np.exp(-abs(z-z0)/h2)
    halodens= dens0*(((r0/(r**2+(z/qh)**2)**0.5))**nh)
    
    return thindens+ftd*thickdens+fh*halodens


@numba.jit
def custom_volume_correction(c, dmin, dmax, nsamp=100):
    """
    A volume correction term that only uses numpy array for speed
    All units are in pc
    """
    dds = np.linspace(dmin,dmax,nsamp)
    r, z=convert_to_rz(c.ra, c.dec, dds)
    rho=juric_density_function(r, z)
    return integrate.trapz(rho*(dds**2), x=dds)/(dmax**3)


class Pointing(object):
    ## a pointing object making it easier to draw samples
    
    def __init__(self, **kwargs):
        #only input is the direction
        self.coord=kwargs.get('coord', None)
        self._samples={}
        self.survey=kwargs.get('survey', None)
        self.mag_limits=None
        self.dist_limits=None
        self.name=kwargs.get('name', None)
        self.volume=None

    def cdf(self,  dmin, dmax):
        """
        The cumulative distribution function along the line of sight
        """
        @numba.vectorize("float64(float64)")
        def get_cdf_point(x):
            ##get the value of the cdf at a given distance
            return (x**3-dmin**3)*custom_volume_correction(self.coord, dmin,x)
        
        norm=(dmax)**3*custom_volume_correction(self.coord, dmin, dmax)
        dds=np.logspace(np.log10(dmin), np.log10(dmax), 5000)
        cdf=get_cdf_point(dds)
        return dds, cdf/norm

    def compute_distance_limits(self):
        """
        computes distance limits based on limiting mags
        """
        rels=wisps.POLYNOMIAL_RELATIONS
        spgrid=np.arange(20, 38)
        if self.mag_limits is None:
            pass
        else:
            #use F140W for 3d-hst pointing and f110w for wisps
            pol=None
            maglmts=None
            if self.survey=='wisps':
                pol=rels['sp_F110W']
                maglmts= self.mag_limits['F110W']
            if self.survey=='hst3d':
                pol=rels['sp_F140W']
                maglmts=self.mag_limits['F140W']

            #compute asbmags using abolute mag relations
            absmags=pol(spgrid)
            relfaintmags=np.array([maglmts[0] for s in spgrid])
            relbrightmags=np.array([maglmts[1] for s in spgrid])
            
            #compute distances
            dmins=get_distance(absmags, relbrightmags)
            dmaxs=get_distance(absmags, relfaintmags)

            distances=np.array([dmaxs, dmins]).T

            self.dist_limits=dict(zip(spgrid, distances))
            #create a dictionary

    def computer_volume(self):
        """
        given area calculate the volume
        """
        volumes={}
        solid_angle=SOLID_ANGLE
        for k in self.dist_limits.keys():
             vc=spsim.volumeCorrection(self.coord,  self.dist_limits[k][1], self.dist_limits[k][0])
             volumes['vc_'+str(k)]=vc
             volumes[k]= vc*0.33333333333*(self.dist_limits[k][0]**3-self.dist_limits[k][1]**3)

        self.volume=volumes
    

    def random_draw(self,  dmax, dmin, nsample=1000):
        """
        randomly drawing x distances in a given direction
        """
        dvals, cdfvals=self.cdf(dmin, dmax)
        @numba.vectorize("int32(float64)")
        def invert_cdf(i):
            return bisect.bisect(cdfvals, i)
        x=np.random.rand(nsample)
        idx=invert_cdf(x)
        res= np.array(dvals)[idx-1]
        return res

    @property
    def samples(self):
        return self._samples
    
    def create_sample(self, nsample=1000):
        self._samples={}
        for k in  self.dist_limits.keys():
            #draw up to twice the distance limit
            self._samples[k]=self.random_draw( 2*self.dist_limits[k][0], self.dist_limits[k][1], nsample=nsample)
    
      
#simulate spectral types
def simulate_spts(**kwargs):
    """
    simulate a distribution of spectral types using a mass function and evolutionary models
    """
    recompute=kwargs.get('recompute', False)
    
    if recompute:

        norm_range = [0.09,0.1]
        norm_density = 0.0055
        nsim = kwargs.get('nsample', 1e2)
        spts=np.arange(17, 40)

        # simulation
        masses = spsim.simulateMasses(nsim,range=[0.02,0.15],distribution='power-law',alpha=0.5)
        norm = norm_density/len(masses[np.where(np.logical_and(masses>=norm_range[0],masses<norm_range[1]))])


        ages=[]
        teffs=[]
        spts=[]

        #uniform distribution
        ages_unif= spsim.simulateAges(nsim,range=[0.1,10.], distribution='uniform')
        teffs_unif = spev.modelParameters(mass=masses,age=ages_unif, set='baraffe03')['temperature'].value
        spts_unif = np.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True) for x in teffs_unif])

        #rujor
        #rujopakarn
        ages_ruj= spsim.simulateAges(nsim,range=[0.1,10.], distribution='rujopakarn')
        teffs_ruj = spev.modelParameters(mass=masses,age=ages_ruj, set='baraffe03')['temperature'].value
        spts_ruj = np.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True) for x in teffs_ruj])

        #aumer
        ages_aum= spsim.simulateAges(nsim,range=[0.1,10.], distribution='aumer')
        teffs_aum = spev.modelParameters(mass=masses,age=ages_aum, set='baraffe03')['temperature'].value
        spts_aum = np.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True) for x in teffs_aum])




        ages.append(ages_unif)
        spts.append(spts_unif)
        teffs.append(teffs_unif)

        ages.append(ages_ruj)
        spts.append(spts_ruj)
        teffs.append(teffs_ruj)

        ages.append(ages_aum)
        spts.append(spts_aum)
        teffs.append(teffs_aum)


        betas=np.logspace(-2, 1, 10)
        for b in betas:
            ages2=spsim.simulateAges(nsim, beta=b, age_range=[0.1,10.])
            teffs2 = spev.modelParameters(mass=masses,age=ages2, set='baraffe03')['temperature'].value
            spts2 = np.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True) for x in teffs2])

            ages.append(ages2)
            teffs.append(teffs2)
            spts.append(spts2)
        

        
        values={'mass': masses, 'ages':np.array(ages), 'teffs':np.array(teffs), 'spts':np.array(spts), 'norm':norm, 'betas': betas}

        import pickle
        with open(wisps.OUTPUT_FILES+'/mass_age_spcts.pkl', 'wb') as file:
           pickle.dump(values,file)
    else:
        values=pd.read_pickle(wisps.OUTPUT_FILES+'/mass_age_spcts.pkl')


    return values

def drop_nan(x):
    x=np.array(x)
    return x[(~np.isnan(x)) & (~np.isinf(x)) ]

    ##selection function
