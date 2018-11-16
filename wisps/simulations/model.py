# -*- coding: utf-8 -*-

"""
Please ignore this, look at the notebooks instead

"""
from .initialize import *
import astropy
import splat
import splat.empirical as spe
import pandas as pd
#from multiprocessing import Pool
import matplotlib.pyplot as plt
#from itertools import izip
from tqdm import tqdm
from functools import partial
#from mementos import =MementoMetaclass
#from cpython cimport array
#import scipy.integrate as integrate
import line_profiler 
#import array
from wisps.utils import Memoize, profile, memoize_func
from tempfile import mkdtemp
cachedir = mkdtemp()
from joblib import Memory
memory = Memory(cachedir=cachedir, verbose=0)

RSUN=CONSTANTS['RSUN']
ZSUN=CONSTANTS['ZSUN']
N_0=CONSTANTS['N_0']

#load all absolute mags and and spectral types #for speed up (splat is slow sometimes)
#SPTS= [splat.typeToNum(x) for x in np.arange(15, 40)]
#ABS_MAGS=np.array([spe.typeToMag(x,'2MASS J',unc=0.5) for x in SPTS])

#MAGS_DICT= pd.DataFrame()
#MAGS_DICT['J']=ABS_MAGS[:,0]
#MAGS_DICT['J_ER']=ABS_MAGS[:,1]
#MAGS_DICT['SPTS']=SPTS
#print(MAGS_DICT)

splat.initializeStandards()

#################################################################################################
class Galaxy(object):
    """
    This is a galaxy model
    """
    def __init__(self, **kwargs):
        """
        Initialization of model parameters
        """
        self._constants=CONSTANTS
        self.spt_range=['M8', 'T9']
        self._parameters=kwargs.get('parameters', {'halo_params': None,
        											'thick_d_params': None,
											'thin_d_params':None})
        self._pointings=kwargs.get('pointings', None)
        self.saturation_limit={'J':14.0} #random, should check the survey
    
    def __repr__(self):
        return "galaxy model "
    
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, new_parameters):
        z = self._parameters.copy()   
        z.update(new_parameters) 
        self._parameters=z
    
    @staticmethod
    def get_galactic_coords(ra, dec, distance):
        """
        distance must be in parsecs
        """
        coords= astropy.coordinates.SkyCoord(ra=ra ,dec=dec, distance=distance)
        
        galoc = astropy.coordinates.Galactocentric(x=coords.cartesian.x,
                           y=coords.cartesian.y,
                             z=coords.cartesian.z,
                              z_sun=ZSUN, galcen_distance=ZSUN)
        return galoc
    
    
    @staticmethod
    @memory.cache
    def bin_densities(dens, dds, **kwargs):
        """
        bins densities and sum them at the same distance"""
        
        bins=kwargs.get('bins', np.logspace(0, 5, num=50) )
        def bint(bins, x,y):
            return sum_bins(x.flatten(), bins,y.flatten())
        total=[bint(bins, x, y) for x, y in zip(dens, dds)]
        return (bins, np.nansum(total, axis=0))
        
        
    def halo_density_profile(self, r, z):
        """
        Disk density up to a distance (r, z)
        """
        f_h, kappa,p=self._parameters['halo_params']
        return f_h*((r**2+z**2)**(p/2.0))
    
    
    def thin_disk_density_profile(self, r, z):
        """
        Disk density up to a distance (r, z)
        """
        hr,hz=self._parameters['thin_d_params']
        return N_0.value*np.exp((-r)/hr)*np.exp((-abs(z))/hz)
    
    def thick_disk_density_profile(self, r, z):
        """
        Disk density up to a distance (r, z)
        """
        hr, hz, fr=self._parameters['thick_d_params']
        return fr*N_0.value*np.exp((-r)/hr)*np.exp((-abs(z))/hz)
    
    @property
    def raw_density(self):
        """
        unbinned densities 
        """
        #sum over all spectral types
        thindens=np.array(list(self.parameters['thin_d_density'].values()))
        thickdens=np.array(list(self.parameters['thin_d_density'].values()))
        halodens=np.array(list(self.parameters['halo_density'].values()))
        #sum over all fields
        sum1=np.nansum( thindens, axis=0)
        sum2=np.nansum(thickdens, axis=0)
        sum3=np.nansum(halodens, axis=0)
        #sum over all fields and spectral types
        return {'thin_disk':sum1, 'thick_disk': sum2, 'halo': sum3}
    
    @property
    def density(self):
        """
        binned densities
        """
        thick_dens=[]
        thin_dens=[]
        halo_dens=[]

        return {'thin_disk':None, 'thick_disk':None ,
                'halo':None, 'total':None, 'distance':None}
    

    @property
    def distance_bins(self):
        return self.parameters['distance']
    
    @property
    def exctinction(self, ra, dec):
        """
        """
        return 0.0
    
    @property
    def pointings(self):
        """
        ras and decs and limiting mags of pointings in the galaxy that I should care about 
        pointings muts be dictionaries with lists of ras, decs, limiting mags, areas
        """
        return self._pointings
    
    @pointings.setter
    def pointings(self, new_pointings):
        field_densities=[]
        for pnting in tqdm(np.array(new_pointings).T):
             ra, dec, limit_mag, area=pnting
             #print (ra, dec, limit_mag)
             field_densities.append( self.integrate_pointing( ra*u.degree, dec*u.degree, limit_mag, area))
        
        #update parameters with the results
        res_copy=  self._parameters.copy()   
        res_copy.update((pd.DataFrame(field_densities).to_dict())) 
        self._parameters=res_copy
        self._pointings=new_pointings
    
    def integrate_pointing(self, ra, dec, limiting_mag, area):
        """
        integrate up to a distance for all spectral types for one poitings
        """
        spts=np.arange(splat.typeToNum(self.spt_range[0]), splat.typeToNum(self.spt_range[1]))
        #convert J and H into 16O and 140 for each spctral type
        abs_mags=MAGS_DICT[MAGS_DICT.SPTS.isin([splat.typeToNum(x) for x in spts])].as_matrix()
        d_near=(np.array([10**((self.saturation_limit['J']+5.0-M)/5.0) for M in abs_mags[:,0]]))*u.pc
        d_far=(np.array([10**((limiting_mag['J']+5.0-M)/5.0) for M in abs_mags[:,0]]))*u.pc
        #make the distance galacto-centric
        results=[]
        #
        nsteps=5000
        dds= np.logspace(-1, 5 , nsteps)
        rrs= np.logspace(-1, 5 , nsteps)
        zzs= np.logspace(-1, 5 , nsteps)
        for  dn, dfar, jmag, spt in zip(d_near, d_far, abs_mags[:,0], spts):
            result=self.integrate(dds, rrs, zzs, ra, dec, float(jmag), dn.value, dfar.value, area)
            result['SpT']=spt
            results.append(result)
 
        integrate_results=pd.DataFrame(results).to_dict('series')
        integrate_results['ra']=ra.value
        integrate_results['dec']=dec.value
        integrate_results['area']=area.value
        #print (integrate_results)
        return integrate_results
    
    #@profile    
    #@Memoize
    def integrate(self,dds, rrs, zzs, ra, dec, jmag,  near_distance, far_distance, area):
        """
        integrate the luminosity functions up to some distances
        """
        #get the luminosity function
        phi=self.luminosity_function(jmag)#*1000.0
        
        r_near, theta_near, z_near= conv_to_galactic(ra.value, dec.value,  near_distance)
        r_far, theta_far, z_far= conv_to_galactic(ra.value, dec.value,   far_distance)
       
        #distance
        dist=far_distance-near_distance
        #divide the volume into 50 little trapezoidals
        #convert area to float
        area=area.to(u.arcsec**2).value
        #create pyramid volumes
        volumes=volume_bins(ra, dec, dist, dds, area)
        volumes=np.array(abs(volumes[0]))
        #solid angle
        #omega=area/(dist**2.0)
        #compute density profiles
        
        #forget about the integrals 
        #only compute densities where the function is in the the range
        dds[((dds<near_distance) | (dds>far_distance))]=np.nan
        rrs[((rrs<r_near) | (rrs>r_far))]=np.nan
        zzs[((zzs<z_near) | (zzs>z_far))]=np.nan
        
        halo_n=self.halo_density_profile(rrs, zzs)*phi#*omega
        thin_disk_n=self.thin_disk_density_profile(rrs, zzs)*phi#*omega
        thick_disk_n=self.thick_disk_density_profile(rrs, zzs)*phi#*omega
        
        n=thick_disk_n+halo_n+thick_disk_n
       

        return {'number': n*volumes, 'luminosity':phi, 'total_density': n,
        	'thin_d_density':thin_disk_n,
        	'thick_d_density':thick_disk_n,
        	'halo_density':halo_n,
               'distance':dds, 'volumes': volumes, 'zs': zzs, 'rs':rrs, 'ra_dec':[ra.value, dec.value]}
        
        
    @staticmethod
    def luminosity_function(J):
        power=-0.3+0.11*(J-14)+0.15*(J-14.0)**2 + 0.015*(J-14.0)**3-0.0002*(J-14)**4
        return 10**power        


   
    def plot(self):
        return


class MissingJMagnitudeError(Exception):
    pass
    
######################################################
#########################################################
@memoize_func
def sum_bins(arr, bins, ref):
            res=[np.nanmean(arr[list(np.array(np.where((ref>bins[i]) & (ref <=bins[i+1]))))]) for i in range(0,len(bins)-1)]
            prev_val=res[0]
            res=np.insert(res, 0, prev_val, axis=0)
            return np.nan_to_num(res)
@memoize_func       
def replace_nans_with_previous_value(a):
            for i in range(0, len(a)): 
                if (np.isnan(a[i])) or (a[i]==0.0): a[i]=a[i-1]
            return a
@memoize_func
def volume_bins(ra, dec, dist, dds, area):
    """
	returns volume bins up to a distance 
	given a delta distance
    """
    area=area*(u.arcmin**2).to(u.arcsec**2)
    print (area)
    return  1/3. * ( ( dist + dds ) ** 3 -  dist** 3) * \
                      ( np.cos( ( np.pi/2 - dec.to(u.radian).value -  area / 2. ) ) -np.cos( ( np.pi/2 - dec.to(u.radian).value+ area / 2.)) ) * \
                      ( ( ra.to(u.radian).value + area/ 2. ) - ( ra.to(u.radian).value - area / 2. ) )
@memoize_func              
def conv_to_galactic(ra, dec, d):
    '''
    Function to convert ra, dec, and distances into 
    Galactocentric coordinates R, theta, Z.
    From Loki
    '''

    const = pd.Series(CONSTANTS)
    r2d   = 180. / np.pi # radians to degrees

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray)  == False:
        ra  = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
    if isinstance(d, np.ndarray)   == False:
        d   = np.array(d).flatten()

    # Convert values to Galactic coordinates
    """
    # The SLOOOOOOOOOOW Astropy way
    c_icrs = SkyCoord(ra = ra*u.degree, dec = dec*u.degree, frame = 'icrs')  
    l, b = c_icrs.galactic.l.radian, c_icrs.galactic.b.radian
    """
    l, b = radec2lb(ra, dec)
    l, b = np.deg2rad(l), np.deg2rad(b)
    
    r    = np.sqrt( (d * np.cos( b ) )**2 + const.RSUN.value * (const.RSUN.value - 2 * d * np.cos( b ) * np.cos( l ) ) )
    t    = np.rad2deg( np.arcsin(d * np.sin( l ) * np.cos( b ) / r) )
    z    = const.ZSUN.value + d * np.sin( b - np.arctan( const.ZSUN.value / const.RSUN.value) )
    
    return r, t, z
#################################################
@memoize_func
def radec2lb(ra, dec):

    '''
    Convert ra,dec values into Galactic l,b coordinates
    '''

    # Make numpy arrays
    if isinstance(ra, np.ndarray)  == False:
        ra = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
        
    # Fix the dec values if needed, should be between (-90,90)
    dec[ np.where( dec > 90 )]  = dec[ np.where( dec > 90 )]  - 180
    dec[ np.where( dec < -90 )] = dec[ np.where( dec < -90 )] + 180
    
    psi    = 0.57477043300
    stheta = 0.88998808748
    ctheta = 0.45598377618
    phi    = 4.9368292465
    
    a    = np.deg2rad( ra ) - phi
    b    = np.deg2rad( dec )
    sb   = np.sin( b )
    cb   = np.cos( b )
    cbsa = cb * np.sin( a )
    b    = -1 * stheta * cbsa + ctheta * sb
    bo   = np.rad2deg( np.arcsin( np.minimum(b, 1.0) ) )
    del b
    a    = np.arctan2( ctheta * cbsa + stheta * sb, cb * np.cos( a ) )
    del cb, cbsa, sb
    ao   = np.rad2deg( ( (a + psi + 4. * np.pi ) % ( 2. * np.pi ) ) )

    return ao, bo

################################################################################
@memoize_func
def build_a_model(theta,  fields, **kwargs):
    """
    build a galaxy model given parameters
    input must only be python lists or floats
    
    """
    model=Galaxy()
    thin_h_z, thin_h_r, thick_fr, thick_h_z, thick_h_r, halo_fr, kappa, p= theta 
    ##
    #print ('params {}'.format(theta))
    model.parameters={'halo_params':[halo_fr, kappa,p],
                      'thick_d_params':[thick_h_r,thick_h_z, thick_fr],
                      'thin_d_params':[thick_h_r,thick_h_z]}

    
    model.pointings = fields
    return model