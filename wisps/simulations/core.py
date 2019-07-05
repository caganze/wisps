

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
        self._samples=[]

    def cdf(self,  dmin, dmax):
        """
        The cumulative distribution function along the line of sight
        """
        @numba.vectorize("float64(float64)")
        def get_cdf_point(x):
            ##get the value of the cdf at a given distance
            return (x**3-dmin**3)*custom_volume_correction(self.coord, dmin,x)
        
        norm=6*(dmax**3)*custom_volume_correction(self.coord, dmin, 2*dmax)
        dds=np.linspace(dmin+1.0, dmax, 200)
        cdf=get_cdf_point(dds)
        return dds, cdf/norm
    

    def random_draw(self,  dmin, dmax, nsample=1000):
        """
        randomly drawing x distances in a given direction
        """
        dvals, cdfvals=self.cdf(dmax, dmin)
        @numba.vectorize("int32(float64)")
        def invert_cdf(i):
            return bisect.bisect(cdfvals, i)-1
        x=np.random.rand(nsample)
        idx=invert_cdf(x)
        return np.array(dvals)[idx]

    @property
    def samples(self):
        return self._samples
    
    def create_sample(self, dmin, dmax):
        self._samples.append(self.random_draw(dmax, dmin, nsample=10000))
    
      
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

#compute effective volumes
def compute_effective_volumes(**kwargs):
    """
    compute effective volumes for all the pointings in my simulation
    """
    recompute=kwargs.get('recompute', False)
    bmags=kwargs.get('bmags', {'F110W': 18.0, 'F140W':18.0, 'F160W':18.0})
    fmags=kwargs.get('fmags', {'F110W': 22.5, 'F140W':22.6, 'F160W':22.7})
    
    spgrid=np.arange(20, 38)

    if recompute:
        area=AREA
        solid_angle=SOLID_ANGLE
        rels=POLYNOMIAL_RELATIONS
        coords=OBSERVED_POINTINGS
        ds=[]
        for spt in spgrid:
               dmax=None
               dmin=None
               pol=rels['sp_F140W']
               absf140=pol(spt)
               dmax=(10.**(-(absf140-fmags['F140W'])/5. + 1.))
               dmin=(10.**(-(absf140-bmags['F140W'])/5. + 1.))
               print ('spt {} distance {}'.format(spt, dmax-dmin))
               ds.append([dmin, dmax])

        vols=[]
        vcs=[]
        for coord in tqdm(coords):
           vs=[]
           vcor=[]
           for d in ds:
               vc=spsim.volumeCorrection(coord, d[0], d[1])
               vs.append([vc*0.33333333333*solid_angle*(d[1]**3-d[0]**3)])
               vcor.append(vc)
           vols.append(vs)
           vcs.append(vcor)

        import pickle
        with open(wisps.OUTPUT_FILES+'/volumes.pkl', 'wb') as file:
           pickle.dump([ds, spgrid, vols, vcs], file)

    else:
        ds, spgrid, vols, vcs=pd.read_pickle(wisps.OUTPUT_FILES+'/volumes.pkl')

    return  ds, spgrid, vols, vcs

def drop_nan(x):
    x=np.array(x)
    return x[(~np.isnan(x)) & (~np.isinf(x)) ]

    ##selection function

    
@numba.vectorize("float64(float64, float64)")
def probfunction(x, y):
    # a custom version of my selection function
    return sf.probability_of_selection(x, y)


def compute_effective_numbers(spts, dists, spgrid):
    ##given a distribution of masses, ages, teffss
    ## based on my polynomial relations and my own selection function
    def match_dist_to_spt(spt, distances):
        """
        one to one matching between distance and spt
        to avoid all sorts of interpolations or binning
        """
        try:
            indexs=np.arange(0, len(spgrid))
            idx=(indexs[(spgrid==round(spt))])[0]
            return random.choice(distances[idx])
        except IndexError:
            return np.nan

    #polynomial relations
    rels=POLYNOMIAL_RELATIONS
    #effective volumes
    dlimits, spgrid, vols, vcs=compute_effective_volumes()
    #assign distances
    dists_for_spts=np.array(list(map(lambda x: match_dist_to_spt(x, dists), spts)))
    #compute magnitudes absolute mags
    f110s= rels['sp_F110W'](spts)
    f140s= rels['sp_F140W'](spts)
    f160s= rels['sp_F160W'](spts)
    #compute apparent magnitudes
    appf140s=f140s+5*np.log10(dists_for_spts/10.0)
    #compute snr based on my relations
    #only use F140W for SNRJS
    #offset them by the scatter in the relation
    f140_snrj_scatter=rels['sigma_log_f140']
    snrjs=10**np.random.normal(np.array(rels['snr_F140W'](appf140s)), f140_snrj_scatter)
    #apply the selection function (this is the slow part)
    sl=probfunction(spts, snrjs)

    #group these by spt
    df=pd.DataFrame()
    df['spt']=spts
    df['ps']=sl
    df['appF140']=appf140s
    df['snr']=snrjs
    #round the spt for groupings
    df.spt=df.spt.apply(round)

    #make selection cuts 
    df_cut=df.drop(df[(df.appF140>22.6) & (df.appF140> 18.0) & (df.snr > 3.0)].index)

    #group by spt and sum
    phi0=[]
    phi0_spts=[]

    for g in df_cut.groupby('spt'):
        phi0_spts.append(g[0])
        phi0.append(np.nansum(g[1].ps))

    idx=[i for i, x in enumerate(phi0_spts) if x in spgrid]

    #finally some luminosity function
    phi=np.array(phi0)[idx]
    #return all the data

    return f110s, f140s, f160s, dists_for_spts, appf140s, snrjs, phi




def compute_effective_distances(**kwargs):
  """
  This is embarrassingly parallel

  """
  recompute=kwargs.get('recompute', False)
  nsamples=kwargs.get('nsamples', 10)
  if recompute:
    dlimits, spgrid, vols, vcs=compute_effective_volumes()
    pnts=[Pointing(coord=x) for x in OBSERVED_POINTINGS]

    list(map(lambda x: [x.create_sample(0.5*dlim[0], 2.0*dlim[1]) for dlim in dlimits], pnts))
    import pickle
    with open(wisps.OUTPUT_FILES+'/eff_distances.pkl', 'wb') as file:
           pickle.dump(pnts, file)
  else:
    pnts=pd.read_pickle(wisps.OUTPUT_FILES+'/eff_distances.pkl')
  return pnts
