
# -*- coding: utf-8 -*-
"""
I use this to maniuplate images of the sources, 
extracted from a larger file
"""
from astropy.wcs import WCS
from astropy.io import fits
import glob
from .initialize import *

class Image(object):
    """
    This heps create image cutouts in large mosaics
    """
    def __init__(self, **kwargs):
        self._name=kwargs.get('name', None)
        self._coords=None
        self._ra=None
        self._dec=None
        self._f110=None
        self._f140=None
        self._f160=None
        self.survey=None
        self._phot_img_pixel=40.0
        self.path=None
        if 'name' in  kwargs: self.name=kwargs.get('name')
        
    def __repr__(self):
        if self._name is None:
            return 'Empty image'
        else:
            return 'Image of '+ self._name
            
    @property
    def name(self):
    	return self._name
    
    @name.setter
    def name(self, new_name):
    	#get images
    	self._name=new_name
    	self._f110=self._grab_object_from_field_image('F110')
    	self._f140=self._grab_object_from_field_image('F140')
    	self._f160=self._grab_object_from_field_image('F160')
    	
    @property
    def ra(self):
    	return self._ra

    @property
    def dec(self):
    	return self._dec
    	
    @property
    def f140(self):
    	"""
    	image in the F140W filter
    	contains: grid, centroid and image data
    	"""
    	return self._f140
    
    @property
    def f110(self):
    	"""
    	image in the F110W filter
    	"""
    	return self._f110
    
    @property
    def f160(self):
    	"""
    	image in the F160W filter
    	"""
    	return self._f160
    
    @property
    def grid(self):
    	"""
    	grid onto which the image is plotted
    	"""
    	return self._grid
    
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
        #rescale the image
        self._f110=self._grab_object_from_field_image('F110')
        self._f140=self._grab_object_from_field_image('F140')
        self._f160=self._grab_object_from_field_image('F160')
    	
    def _grab_object_from_field_image(self, filt, **kwargs):
        """
        This function grabs an image given ra, dec and a filter
		filter must be either "F140", "F160", "F110"
		
		returns:
		the grid onto which the object should be plotted
		the image data
        the pixel positions of the object
        """
        #create filename of the field image
        n=self.name
        if n.lower().startswith('par'):
            #p=REMOTE_FOLDER+'wisps/'+n.split('_')[0]+'_*/DATA/DIRECT_GRISM/'+filt+'*_sci.fits'
            p=REMOTE_FOLDER+'wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+n.split('-')[0]+'/hlsp_wisp_hst_wfc3_*-80mas_f*w_v6.2_drz.fits'
            #print ('glob', glob.glob(p), p)
        if n.lower().startswith('uds') or n.lower().startswith('aeg') or n.lower().startswith('cos') :
            p=REMOTE_FOLDER+n.split('-')[0]+'*/'+n.split('-G')[0]+'/'+n.split('-G')[0]+'*'+filt+'*drz_sci.fits'
        if n.lower().startswith('goo'):
        	p=REMOTE_FOLDER+'goods'+'*/'+n.split('-G')[0]+'/'+n.split('-G')[0]+'*'+filt+'*drz_sci.fits'
        #check filter
        valid_filters=['F140', 'F110', 'F160']
        if filt not in valid_filters:
            raise InvalidFilterError('Filter {} is not valid, filter options: {}'.format(filt,valid_filters))
            
        #if the field was not imaged in given filter 
        filter_file=glob.glob(p)
        self.path=p
        #print (p)
        try:
            #print ('i guess not ? wtf  {}'.format(filter_file))
            #create wcs object
            w1=WCS(filter_file[0])
            
            #read the file
            if n.lower().startswith('par'):
                with fits.open(filter_file[0], memmap=False) as hdu:
                    t=hdu[1].data
            else:
                with fits.open(filter_file[0], memmap=False) as hdu:
                    t=hdu[0].data
        
            pixelsize=self.pixels_per_image
            #get pixel position of the center object
            
            #get pixel positions from ra and dec
            py0, px0 = w1.wcs_world2pix(self.ra, self.dec, 1)
            #print (py0, px0)
            px0=abs(px0)
            py0=abs(py0)
            
            #grab a box around it
            px1, py1 = np.array([px0, py0])-pixelsize
            px2, py2 = np.array([px0, py0])+pixelsize
            
            #not getting outside the bounds of the image
            bools=[bool(int(px0-pixelsize) <= 0),bool(int(py0-pixelsize) <= 0),
                   bool(int(px0+pixelsize) >= t.shape[0]),   bool(int(py0+pixelsize) >= t.shape[1])]
            #print (bools)
            if bools[0]:px1=0
            if bools[1]:py1=0
            if bools[2]:px2=t.shape[0]
            if bools[3]:py2=t.shape[1]
                
            #slicing the data
            data  = t[slice(int(px1), int(px2)), slice(int(py1), int(py2))]
            #creating a grid for plotting
            grid   = np.mgrid[slice(int(px1), int(px2)), slice(int(py1), int(py2))]
            
            #if everything is outside the bounds
            if np.all(bools): 
                data=t
                grid=np.mgrid[0:t.shape[0]:1, 0:t.shape[1]:1]
            
            output={'filter':filt+'W',
    				'data': data,
    				'center': (px0, py0),
    				'grid':grid,
    				'is_white':False}
            del t
            hdu.close()
            del hdu
        except:
            #make a white image if everything else fails
            white_img = np.zeros([100,100,3],dtype=np.uint8)
            white_img.fill(255)
            output= {'filter':filt,'data': white_img,'center': (0.0, 0.0),'grid':None, 'is_white': True}
            
            
        return output

class InvalidFilterError(Exception):
	pass

