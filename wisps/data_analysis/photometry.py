

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the main module for  adding photometry information given a spectrum(grism id)


@author: caganze


"""
from .initialize import *
from astropy.io import ascii
from astropy.table import hstack, Table, vstack
import glob
from .spectrum_tools import *
from .image import*
from .plot_spectrum import plot_source
from ..utils.tools import get_distance
import splat.core as spl
import splat.empirical as spe
import random
from astropy.coordinates import SkyCoord
import copy
import wisps

from functools import lru_cache

from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial

class Source(Spectrum):
    """
    Source object, inherts from Spectrum but adds photometry and distances
    
    Attributes:
        name (str): grism id in the photometric catalog
        shortname (str): a shortname based on the coordinates
        wave (numpy array): the wavelength array i.e for flux, contam, noise
        indices (dict): dictionary of spectral indices i.e ratio of fluxes 
        cdf_snr (str): modified snr 
        coords (object): astropy.coordinates of the source
        mags (dict): dictionary of mags in different filters
        distance (dict): dictionary of distances estimated from  different filters
        splat_spectrum: a splat spectrum object with all its glory

    Example:
        >> import wisps
        >> s=wisps.Source(name='uds-25-G141_36758')
        >> s.plot()
        >> print ('mags {} distance {} coords {}'.format(s.mags, s.distance, s.coords))
    """
    def __init__(self, **kwargs):
        self._coords=None
        self._ra=None
        self._dec=None
        self._mags=None
        self._distance=None
        self._spectrum=None
        self.designation=None
        self._shortname= None
        self._image=None
        self._flags=None
        self._phot_img_pixel=kwargs.get('pixel_per_img', 100)

        super().__init__(**kwargs)
    
        if 'mags' in kwargs:
            self.mags=kwargs.get('mags')
            
        if self._distance is None: self._calculate_distance()
        if self._filename is not None: self.name=kwargs.get('filename', self._filename)

        self.original = copy.deepcopy(self)

    def __repr__(self):
        if self._filename is None:
            return 'anon spectrum'
        else:
            return self._filename
    
    #to do : ras, decs & should stuff probably be one property (e.g a big dictionary) 
    #for simplicity and elegance?
    @property
    def ra(self):
        return self._ra*u.degree
	
    @ra.setter
    def ra(self, new_ra):
        self._ra= new_ra
        
	
    @property
    def dec(self):
        return self._dec*u.degree
        
    @dec.setter
    def dec(self, new_dec):
        self._dec=new_dec
       
            
    @property
    def coords(self):
        """
        This must be an astropy skycoord object
        """
        if self._distance is None: self._calculate_distance()
        return self._coords
        
    @coords.setter
    def coords(self, new_coords):
        self._coords=new_coords
        self.ra=new_coords.ra
        self.dec=new_coords.dec
        self.designation=splat.coordinateToDesignation(new_coords)
    
    @property
    def name(self):
        return self._filename
    
    @name.setter
    def name(self, new_name):
        """
        setting up a source object by searching its grism id throughout the master table
        """
        self._filename=new_name
        if new_name.endswith('.ascii'): new_name=new_name.split('.ascii')[0].split('.')[0]
        if  new_name.endswith('.dat'): new_name=new_name.split('.dat')[0]
        if new_name.endswith('.1D'): new_name=new_name.split('.1D')[0]
		
        self.filename=new_name
        #get mags from the photometry catalog
        #this is the master table that contains everything i.e after the snr cut 
        # to see how this is created look at pre_processing.py
        df=COMBINED_PHOTO_SPECTRO_DATA
        s=df.loc[df['grism_id'].apply(lambda x: x.lower()).isin([new_name.lower()])].reset_index().iloc[0]
        
        self._mags={'F110W': (s.F110[0], s.F110[1]), 
                             'F160W': (s.F160[0], s.F160[1]),
                             'F140W': (s.F140[0], s.F140[1])}

        #replace --99 by nan
        for k, val in self._mags.items():
            if (val[0]<0. or val[1] <0.): self._mags[k]=(np.nan, np.nan)
           
        self.coords=SkyCoord(ra=s.RA, dec=s.DEC, unit='deg')
        #self.spectral_type=s.Spts
        self._flags=s['class_star']
        
        #populate image data 
        img=Image(is_ucd=self.is_ucd)
        img._ra=s.RA # i shouldn' be doing this :(
        img._dec=s.DEC
        img.name=new_name
        img.pixels_per_imagep=self.pixels_per_image
        self._image=img
        self.original = copy.deepcopy(self)
        self._calculate_distance()
    
        
        
    @property
    def shortname(self):
    #using splat tricks to create shortnames
        if (self.name is not None ) and not np.isnan(self.ra):
            if self.name.lower().startswith('par'):
                self._shortname=self.designation.replace('J', 'WISP J')
                
            elif self.name.lower().startswith('aegis'):
                self._shortname=self.designation.replace('J', 'AEGIS J')
                
            elif self.name.lower().startswith('goodss'):
                self._shortname=self.designation.replace('J', 'GOODSS J')
            
            elif self.name.lower().startswith('goodsn'):
                self._shortname=self.designation.replace('J', 'GOODSN J')
                
            elif self.name.lower().startswith('uds'):
                self._shortname=self.designation.replace('J', 'UDS J')
                
            elif self.name.lower().startswith('cosmos'):
                self._shortname=self.designation.replace('J', 'COSMOS J')

        return self._shortname
    
    #again, mags and distance should probably be one property called parameters or something 
        
    @property
    def mags(self):
        """
        magnitudes dictionary
        """
        
        return self._mags
    
    @mags.setter
    def mags(self, new_mags):
        self._mags=new_mags
        
    
    @property
    @lru_cache(maxsize=128)
    def distances(self):
        """
        distances measured using absmag/spt relations from Dupuy et al
        """
        #return self._distance
        if self._distance is None:
            self._calculate_distance()
            
        return self._distance
    
    @property
    #@lru_cache(maxsize=128)
    def distance(self):
        """
        Overall distance of the source obtained by averaging all the photomteric distances
        """
        #print (self.distances)
        if self._distance is None:
            self._calculate_distance()

        ds=np.array([self.distances[k] for k in self.distances.keys() if ('dist'  in k) and ('dist_er'  not in k)])
        ers=np.array([self.distances[k] for k in self.distances.keys() if 'dist_er'  in k])

        #distance is the weighted mean and std 
        nans=np.isnan(ds)
        val, unc=spl.weightedMeanVar(ds[~nans], ers[~nans])
        #dont forget the other uncertainties
        unc_tot=(unc**2+(ers[~nans]**2).sum())**0.5
        return {'val':val*u.pc, 'er':unc_tot*u.pc}
    
    @property
    #@lru_cache(maxsize=128)
    def photo_image(self):
    	return self._image
    
    @property
    #@lru_cache(maxsize=128)
    def pixels_per_image(self):
        """
        The number of pixels around the object to show in photometry image
        """
        return self._phot_img_pixel
    
    @pixels_per_image.setter
    def pixels_per_image(self, new_pixels):
        """
        The number of pixels around the object to show in photometry image
        """
        self._phot_img_pixel=new_pixels
        self.photo_image.pixels_per_image=new_pixels

    @property
    def flags(self):
        return self._flags
    
    def _calculate_distance(self):
        
        """
        computing a photo-spectrometric distance of the source 
        
        The spectrum is classified by standard, 
        I compute a color from the standard and obtain an apparent magnitude in the respective filter
        """
        if self.mags is None: return None
    
        if self.spectral_type is None:
            self.spectral_type = splat.classifyByStandard(self.splat_spectrum, comprange=[[1.2, 1.6]], dwarf=True,subdwarf=False,  statistic='chisqr') [0]
            
        
        self._distance= distance(self.mags, self.spectral_type[0],  self.spectral_type[1])
        if self._distance is not None:
            self.coords=SkyCoord(ra=self._ra, dec=self._dec,  distance=self.distance['val'].value*u.pc)
        
        return 
        
   
        
    def plot(self, **kwargs):
        """
        Plotting routine, overwriting Spectrum.plot
        This routine includes a 2d image superimposed to the spectrum
        
        Arguments: 
        
        optional arguments:
    
        """
        plot_source(self, **kwargs)


def getter_function_source(filename):
    """
    partial does not use kwargs so our options are limitted here
    """
    try:
        return Source(filename=filename, is_ucd=False)
    except:
        print ('yikes .....  {}'.format(filename))
        return 

def getter_function_spectrum(filename):
    """
    partial does not use kwargs so our options are limitted here
    """
    return Spectrum(filename=filename, is_ucd=False)


def get_multiple_sources(filenames, **kwargs):
    """
    Load multiple sources at once using multprocessing

    Parameters:
        filenames=filenames
        kwargs: keyword arguemnts
    """
    source_type=kwargs.get('source_type', 'source')

    if source_type=='spectrum':
        method=partial(getter_function_spectrum)

    if source_type=='source':
        method=partial(getter_function_source)

    iterables=[filenames]

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=20))

    results=[x for x in futures]

    return results







