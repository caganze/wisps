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
from .spectrum_tools import*
from .image import*
from .plot_spectrum import plot_source
from ..utils.tools import distance
import splat.core as spl
import splat.empirical as spe
import random
from astropy.coordinates import SkyCoord
import copy


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
        super().__init__()
        self._wisp_name=kwargs.get('name', None)
        self._wisp_name=kwargs.get('filename', None)
        self._coords=None
        self._ra=None
        self._dec=None
        self._mags=None
        self._distance=None
        self._spectrum=None
        self.spectral_type=kwargs.get('spectral_type', None)
        self.designation=None
        self._star_flag=kwargs.get('is_star', True)
        self._shortname= None
        self._image=None
        self._flags=None
        self._phot_img_pixel=kwargs.get('pixel_per_img', 100)
        if 'mags' in kwargs:
            self.mags=kwargs.get('mags')
        if 'name' in kwargs:
            self.name=kwargs.get('name')
            
        if self._star_flag and  self._distance is None: self._calculate_distance()
        self.original = copy.deepcopy(self)

    def __repr__(self):
        if self._wisp_name is None:
            return 'anon spectrum'
        else:
            return self._wisp_name
    
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
        if self._star_flag and  self._distance is None: self._calculate_distance()
        return self._coords
        
    @coords.setter
    def coords(self, new_coords):
        self._coords=new_coords
        self.ra=new_coords.ra
        self.dec=new_coords.dec
        self.designation=splat.coordinateToDesignation(new_coords)
        
    
    @property
    def name(self):
        return self._wisp_name
    
    @name.setter
    def name(self, new_name):
        """
        setting up a source object by searching its grism id throughout the master table
        """
        self._wisp_name=new_name
        if new_name.endswith('.ascii'): new_name=new_name.split('.ascii')[0].split('.')[0]
        if  new_name.endswith('.dat'): new_name=new_name.split('.dat')[0]
        if new_name.endswith('.1D'): new_name=new_name.split('.1D')[0]
		
        self.filename=new_name
        #get mags from the photometry catalog
        #this is the master table that contains everything i.e after the snr cut 
        # to see how this is created look at pre_processing.py
        df=COMBINED_PHOTO_SPECTRO_DATA
       	try:
        	s=df.loc[df['grism_id'].isin([new_name])].reset_index().ix[0]
        except:
        	warnings.warn('This source was removed using snr cut',  stacklevel=2)
        	df=pd.read_hdf(COMBINED_PHOTO_SPECTRO_FILE, key='all_phot_spec_data')
        	s=df.loc[df['grism_id'].isin([new_name])].reset_index().ix[0]
        #print (s)

        self._mags={'F110W': (s.F110[0], s.F110[1]), 
                             'F160W': (s.F160[0], s.F160[1]),
                             'F140W': (s.F140[0], s.F140[1])}
           
        self.coords=SkyCoord(ra=s.RA, dec=s.DEC, unit='deg')
        #self.spectral_type=s.Spts
        self._flags=s['flags']
        
        #populate image data 
        img=Image()
        img._ra=s.RA # i shouldn' be doing this :(
        img._dec=s.DEC
        img.name=new_name
        img.pixels_per_imagep=self.pixels_per_image
        self._image=img
        self.original = copy.deepcopy(self)
        
        #print (self._mags)
        
        
    @property
    def shortname(self):
    #using splat tricks to create shortnames
        if (self.name is not None ) and not np.isnan(self.ra):
            if self.name.lower().startswith('par'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'WISP ')
                
            elif self.name.lower().startswith('aegis'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'AEGIS ')
                
            elif self.name.lower().startswith('goodss'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'GOODSS ')
            
            elif self.name.lower().startswith('goodsn'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'GOODSN ')
                
            elif self.name.lower().startswith('uds'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'UDS ')
                
            elif self.name.lower().startswith('cosmos'):
                self._shortname=spl.designationToShortName(self.designation).replace('J', 'COSMOS ')

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
    def distances(self):
        """
        distances measured using absmag/spt relations from Dupuy et al
        """
        #return self._distance
        if self._distance is None:
            self._calculate_distance()
            
        return self._distance
    
    @property
    def distance(self):
        """
        Overall distance of the source obtained by averaging all the photomteric distances
        """
        #print (self.distances)
        if self._distance is None:
            return self._calculate_distance()
        else:
            ds=[self.distances[k][0] for k in self.distances.keys() if ('dist'  in k) and ('dist_er'  not in k)]
            ers=[self.distances[k] for k in self.distances.keys() if 'dist_er'  in k]

            #print (self._distance)
            #print (ds)
            val=np.nanmean(ds)*u.pc
            unc= np.sqrt(np.nanstd(ds)**2+np.nanmean(ers)**2)*u.pc
            return {'val':val, 'er':unc}
    
    @property 
    def photo_image(self):
    	return self._image
    
    @property
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
            self.spectral_type = splat.classifyByStandard(self.splat_spectrum, comprng=[[1.1, 1.3], [1.3, 1.65]])[0]
            
        
        self._distance= distance(self.mags, self.spectral_type)
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
        

# def __get_photometric_info(sp):
# 	#format catalog names
# 	filters=['F110', 'F140', 'F160']
# 	
# 	if sp.name.startswith('Par'):
# 		n1=s1.name
# 		number=s1.name.split
# 		path_to_cat=wisps.REMOTE_FOLDER+'wisps/'+n1.split('_')[0]+'_*'+'/DATA/DIRECT_GRISM/fin_*'+filt+'.cat'
# 		filter_file=glob.glob(path_to_cat)[0]
# 		cat=ascii.read(filter_file)
# 		ra, dec, mag, mager=__read_phot_catalog( cat, number, survey)
# 		
# 	
# 	return 
# def __read_phot_catalog( cat, number, survey):
# 	"""
# 	gets input catalog and number of the object to extract
# 	return photometry, ra and dec,
# 	
# 	this doens't solve my problem with missing 3d-hst IDS
# 	"""
# 	ra=None
# 	dec=None
# 	mag=None
# 	mager=None
# 	if survey=='wisps':
# 		select=cat[cat.col2==number]
# 		ra=float(select.col1.iloc[0].split('_')[1])
# 		dec=float(select.col1.iloc[0].split('_')[2])
# 		mag=float(select.col13.iloc[0])
# 		mager=float(select.col14.iloc[0])
# 		
# 	if survey=='hst-3d':
# 		select=cat[cat.NUMBER==number]
# 		
# 	return ra, dec, mag, mager
# 	
