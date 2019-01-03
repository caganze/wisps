# -*- coding: utf-8 -*-

"""
Contains the sectrum object
"""


#imports 
##############
from .initialize import *
import splat
from astropy.io import ascii, fits
from matplotlib import gridspec
import glob
import splat.core as spl
from splat import plot as splat_plot
from .indices import measure_indices
from .path_parser import parse_path,get_image_path
import statsmodels.nonparametric.kernel_density as kde
import os
from astropy.visualization import ZScaleInterval
from scipy import interpolate
from scipy import stats

#################
splat.initializeStandards()
###############

class Spectrum(object):
    """
    
    wisps spectrum object
    
    Attributes:
        filename (str): grism id in the photometric catalog
        wave (numpy array): the wavelength array i.e for flux, contam, noise
        indices (dict): dictionary of spectral indices i.e ratio of fluxes 
        cdf_snr (str): modified snr
        splat_spectrum: a splat spectrum object with all its glory
        
    Example:
        >> import wisps
        >> s=wisps.Spectrum(filename='uds-25-G141_36758')
        >> s.plot()
        >> print ('snr {}'.format(s.cdf_snr))
        >> s.splat_spectrum.plot()
    """
    def __init__(self, **kwargs):

        self._wave=kwargs.get('wave', None)
        self._flux=kwargs.get('flux', None)
        self._contam=kwargs.get('contam', None)
        self._noise=kwargs.get('noise', None)
        self._snr=None
        self._empty_flag=None
        self._splat_spectrum=None
        self._filename=None
        self._filepath=None
        self._snr_histogram=None
        self._spectrum_image_path=None
        self.two_d_spectrum=None
        self._survey=None
        self._sensitivity=None
        self._spectrum_image_path=None
        self._indices=None
        self._spectral_type=None

        #load spectrum if given filename 
        
        if 'filepath' in kwargs: self._filepath =kwargs.get('filepath', None)
        if 'filename' in kwargs:  self._filename =kwargs.get('filename', None)
        if 'name' in kwargs: self._filename =kwargs.get('name', None)

        #print (return_path(self._filename))
        if  (self._filename is not None):
            self.filename=self._filename

        if (self._filepath is not None):
        	self.filepath=self._filepath
        	
        if self._wave is not None:
       		 self._compute_snr()
        	 self._splat_spectrum=splat.Spectrum(wave=self._wave, flux=self._flux, noise=self._noise, instrument='WFC3')
    
    def __repr__(self):
        if self._filename is None:return "anon spectrum"
        else: return self._filename
        
    
    @property
    def wave(self):
    	return self._wave
    	
    @property
    def flux(self):
    	return self._flux
    	
    @property
    def contamination(self):
    	return self._contam
    	
    @property
    def noise(self):
    	return self._noise
    
    @property 
    def snr(self):
        return self._snr

    @property
    def empty_flag(self):
        if np.all(self.flux ==-1.0):
            self._empty_flag=True
        if self.flux==np.array([]):
            self._empty_flag=True
        if np.all(self.flux ==0.0):
            self._empty_flag=True
        if np.all(self.flux ==np.nan):
            self._empty_flag=True
            
    @empty_flag.setter
    def empty_flag(self, new_flag):
     	self._empty_flag= bool(new_flag)
     	self._flux=None
     	self._wave=None
     	self._contam=None
     	self._noise=None
     	
    @property 
    def splat_spectrum(self):
    	self._splat_spectrum= splat.Spectrum(wave=self._wave, flux=self._flux, noise=self._noise, instrument='WFC3')
    	return self._splat_spectrum
    
    def classify_by_standard(self, **kwargs):
        """
        Uses splat.classifyByStandard to classify spectra using spex standards
        """ 
        return splat.classifyByStandard(self._splat_spectrum, **kwargs)

    def fit_a_line(self, **kwargs):
        """
        Fit a line, returns a chi-square
        """
        #only fits within the range
        mask=kwargs.get('mask', np.where((self.wave>1.15) & (self.wave <1.65))[0])
        wave=self.wave[mask]
        flux=self.flux[mask]
        noise=self.noise[mask]
        #fit a line from stast linerar regression package
        m, b, r_value, p_value, std_err = stats.linregress(wave, flux)
        line=m*wave+b
        chisqr=np.nansum((flux-line)**2/noise**2)
        #return the line anc chi-square
        return line, chisqr

    def f_test(self, **kwargs):
        """
        Use an F-test to see wether a line fits better than a spectral standard
        """
        #get the splat spectrum
        s=self.splat_spectrum
        #trim within the same wavelength used to compare to standards
        s.trim([1.15, 1.65])
        #fit a line
        line, linechi=self.fit_a_line()
        #compare to standards
        spt, spexchi=splat.classifyByStandard(s, return_statistic=True, fit_ranges=[[1.15, 1.65]], plot=False, **kwargs)
        #calculate f-statistic
        x=spexchi/linechi
        #calculate the f-statistic dfn=2, dfd=1 are areguments
        f=stats.f.pdf(x, 2, 1, 0, scale=1)
        #return result
        result=pd.Series({'spex_chi':spexchi, 'line_chi':linechi, 'spt': spt, 'f':f})
        return result


    def normalize(self, **kwargs):
        """
        :Purpose: Normalize a spectrum to a maximum value of 1 (in its current units)
        :input:
        :ouput:
        :Example:
        """
        sp=self.splat_spectrum
        sp.normalize(**kwargs)
        self._wave= sp.wave.value
        self._flux=sp.flux.value
        self._noise=sp.noise.value
        
        return
        
    @property 
    def cdf_snr(self):
        """
        Returns the snr computed form the SNR kde 
        """
        try:
            sn=np.array(self.flux/self.noise)
            sn=sn[~np.isnan(sn)]
            xgrid=np.linspace(np.nanmin(sn), np.nanmax(sn), len(self.wave))
            cdf=kde_statsmodels_m(sn, xgrid)
            #dirty trick to make sure I always get a value take the biggest value between 80% and 90%
            #should replace this with an interpolotation but it's slower
            #plt.plot(cdf)
            #plt.show()
            sel=np.where(cdf>0.9)[-1] 
            _cdf_snr=xgrid[sel[-1]]
            self._snr_histogram=cdf
        except:
            _cdf_snr=np.nan
			
        return _cdf_snr
    
    def _compute_snr(self):
        """
        different calcultions of snr 
        input:
        output:
        snrs=[]
        """
        #Don't bother for an empty spectrum 
        if self._empty_flag:
            return np.array([np.nan, np.nan, np.nan])
        else:
            #use regions of interest to calculate different snrs
            msk1=np.where((self.wave>=1.2) &(self.wave<=1.3))[0]
            msk2=np.where(((self.wave>=1.2) &(self.wave<=1.3)) | ((self.wave>=1.52) &(self.wave<=1.65)))[0]
            msk3=np.where((self.wave>=1.52) &(self.wave<=1.65) )[0]
            msk4=np.where((self.wave>=1.1) &(self.wave<=1.65) )[0]

            snr1= np.nanmedian(self.flux[msk1]/self.noise[msk1])
            snr2=np.nanmedian (self.flux[msk2]/self.noise[msk2])
            snr3=np.nanmedian (self.flux[msk3]/self.noise[msk3])
            snr4=np.nanmedian (self.flux[msk4]/self.noise[msk4])

            self._snr= {'snr1':snr1, 'snr2':snr2, 'cdf_snr': self.cdf_snr,'snr3':snr3, 'snr4':snr4}
        
    @property
    def indices(self):
        #should use uncertainties
        if self._indices is None: self._indices=measure_indices(self, return_unc=True)
        return self._indices

    def measure_indices(self, **kwargs):
        """
        measure indices of aa spectrum, calls the measureIndicess function
        Takes kwargs for toSplat()
        
        """
        self.normalize()
        return measure_indices(self,**kwargs)

    def add_noise(self, n=1.0,noise=None):
        """
        add n-sigma noise to the spectrum
        """
        mask=np.where((self.wave>1.1) & (self.wave<1.7))[0]
        mu= np.nanmedian(self.noise[mask])
        if noise is not None:
            addn=noise
        else:
            sigma=n*np.nanstd(self.noise[mask])
            addn=np.random.normal(mu,sigma,len(self._flux))
        self._flux=self._flux+addn
        self._noise=self._noise+addn
        self._indices= measure_indices(self, return_unc=True)
        self._compute_snr()

    @property 
    def filepath(self): 
        return self._filepath
    	
    @filepath.setter
    def filepath(self, new_file_path):
        """
        returns a wisp spectrum, given filepath/filename
        """
        #raise error is path does not exist
        #print(new_file_path)
        if not os.path.exists(new_file_path):
            raise NameError('\nCould not find file {}'.format(new_file_path))
        
        data= ascii.read(new_file_path)
        self._filepath=new_file_path
        #this is how I know I have not ran the parser before (I'm trying to avoid duplicating things)
        #ugh this is annoy
        if self._filename is None:
            if not 'wisps' in self._filepath:
                self._filename=self._filepath.split('/')[-1]
            if 'wisps' in self._filepath:
                self._filename=self._filepath.split('/')[-1].split('_wfc3_')[-1].split('a_g102')[0].split('a_g')[0]
            #print (self._filename)
            if self._filename.startswith('hlsp'):
                self._filename=self._filename.split('_wfc3_')[-1].split('a_g102')[0]
            survey, stamp_image_path=get_image_path(self._filename)
            self._spectrum_image_path=stamp_image_path
            self._survey=survey
        
        #print (self._filename)
        #sometimes column keys are "col1', 'col2', 'col3', etc.. instead of wave, flux, error
        try:
            wv= data['col1'] 
            flux=data['col2']
            noise=data['col3']
            contam=data['col4']
        except (ValueError, KeyError):
            wv= data['wave'] 
            flux=data['flux']
            noise=data['error']
            contam=data['contam']
        self._wave=np.array(wv)/10000.0
        
        self._noise=np.array(noise)
        self._contam=np.array(contam)
        # subtract the contamination
        self._flux=np.array(flux)-np.array(contam)
        #add constant to the flux if is negative
        #if np.any(self._flux<0.0):
        if not self._survey == 'wisps':
            with  fits.open(self._spectrum_image_path, memmap=False) as hdu:
                self._sensitivity=hdu[10].data
            #divide by the sensitivity
            self._flux=self._flux/self._sensitivity
            hdu.close()
            del hdu[10].data
        
        #add offset if some of the flux is negative
        #print (self._wave)
        try:
            offset_flux=np.nanmin(self._flux[np.where((self._wave >1.4) & (self._wave <1.5))])
            if offset_flux<0.0:
                     self._flux=self._flux+abs(offset_flux)
        except ValueError: pass
        self._compute_snr()
        self._indices= measure_indices(self, return_unc=True)
        #self._original_flux=self._flux
        #self._original_flux=self._noise
        
    @property 
    def filename(self):
        return self._filename
    @property
    def name(self):
    	return self._filename
    
    @property
    def grism_id(self):
    	return self._filename
    	
    @filename.setter
    def filename(self, new_filename):
        #this is how I know I have not ran the parser before (I'm trying to avoid duplicating things)
        #print ('self filepath', self._filepath )
        if self._filepath is None:
            survey, spectrum_path, stamp_image_path=parse_path(new_filename, 'v5')
            self._filename=spectrum_path.split('/')[-1]
            self._spectrum_image_path=stamp_image_path
            self._survey=survey
            self.filepath= spectrum_path
        else:
            self._filename=new_filename
        
    @property
    def survey(self):
    	return self._survey
        
    @property
    def spectrum_image(self):
        imgdata=None
        if self._survey == 'wisps':
            with fits.open(self._spectrum_image_path, memmap=False) as imghdu:
                imgdata=imghdu[0].data
            return  imgdata
            imghdu.close()
            del mgdata
        else:
            with fits.open(self._spectrum_image_path, memmap=False) as imghdu:
                imgdata=imghdu[5].data-imghdu[8].data
            return imgdata
            imghdu.close()
            del mgdata

    
    @property
    def sensitivity_curve(self):
    	return self._sensitivity
    
    def plot(self, **kwargs):
    	if kwargs.get('with_splat'):
    		splat_plot.plotSpectrum(self._splat_spectrum, **kwargs)
    	else:
    		plot_any_spectrum(self, **kwargs)
    		
    	
    		
def plot_any_spectrum(sp, **kwargs):
    """
	Main plotting tool for a specturm tool, almost a replica of plotting a Source object
    """
    cmap=kwargs.get('cmap', 'viridis')
    compare_to_std=kwargs.get('compare_to_std', False)
    save=kwargs.get('save', False)
    filt=kwargs.get('filter', 'F140W')
    
    #esthetiques
    mask=np.where((sp.wave>1.15)& (sp.wave<1.65))[0]
    xlim= kwargs.get('xlim', [1.15, 1.65])
    xlabel=kwargs.get('xlabel','Wavelength (micron)')
    ylim=kwargs.get('ylim', [np.nanmin(sp.flux[mask]), np.nanmax(sp.flux[mask])])
    ylabel=kwargs.get('ylabel','Normalized Flux')
    
    #paths
    
    #create the grid
    gs = gridspec.GridSpec(2, 3, height_ratios=(1, 3))
    fig=plt.figure(figsize=(8,6))
    ax1 = plt.subplot(gs[0, 0]) 
    ax2 = plt.subplot(gs[0, 1:3]) 
    ax3 = plt.subplot(gs[1, :]) 
    
    #remove markers from images
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    
    l1,=ax3.step(sp.wave, sp.flux, color='k')
    l2,=ax3.plot(sp.wave, sp.noise, 'c')
    try:
        l4, =ax3.plot(sp.wave, sp.contamination, 'g')
    except:
        l4=None
    
    
    plts=[l1, l2, l4]
    
    #compare to standards
    if compare_to_std:
        spectral_type=splat.classifyByStandard(sp.splat_spectrum, fit_ranges=[[1.15, 1.65]])[0]
        std=splat.getStandard(spectral_type)
        chi, scale=splat.compareSpectra(sp.splat_spectrum, std, fit_ranges=[[1.15, 1.65]])
        #std.scale(scale)
        l3,=ax3.step(std.wave, std.flux, color='y')
        plts.append(l3)
    
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    
    ax3.set_xlabel(xlabel, fontsize=18)
    ax3.set_ylabel(ylabel, fontsize=18)
    
    #add the 2d spectrum
    try:
        v0, v1=ZScaleInterval().get_limits(sp.spectrum_image)
        ax2.imshow(sp.spectrum_image, vmin=v0, vmax=v1, cmap='Greys', aspect='auto')
        ax2.set_xlabel('G141', fontsize=15)
    except:
        pass
    
    #fig.legend(tuple(plts))
    if save:
        filename=kwargs.get('filename', OUTPUT_FIGURES+sp.name+'.pdf')
        plt.savefig(filename)
    return 
    		

def kde_statsmodels_m(x, x_grid, **kwargs):
    """
    multivariate kde
    """
    model=kde.KDEMultivariate(x, bw='normal_reference', var_type='c')
    return model.cdf(x_grid)
    
if __name__=='__main__':
	pass
    