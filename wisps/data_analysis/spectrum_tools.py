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
import copy
from ..utils import memoize_func
import numba
from scipy import stats
from ..utils.tools import get_distance, make_spt_number
from ..data_sets import datasets


POLYNOMIAL_RELATIONS= pd.read_pickle(OUTPUT_FILES+'/polynomial_relations.pkl')

from functools import lru_cache #high performance memoization

#################
splat.initializeStandards()
splat.initiateStandards()
splat.initiateStandards(sd=True)
splat.initiateStandards(dsd=True)
splat.initiateStandards(esd=True)
splat.initiateStandards(vlg=True)
splat.initiateStandards(intg=True)

STD_DICTS =splat.STDS_DWARF_SPEX.copy()
STD_DICTS.update(splat.STDS_VLG_SPEX)
STD_DICTS.update(splat.STDS_INTG_SPEX)
STD_DICTS.update(splat.STDS_SD_SPEX)
STD_DICTS.update(splat.STDS_DSD_SPEX)
STD_DICTS.update(splat.STDS_ESD_SPEX)

###############
#@numba.jitclass()
UCD_SPECTRA=datasets['ucd_data']

def interpolated_standards():
    stds=splat.STDS_DWARF_SPEX
    interpstds={}
    for k in stds.keys():
        s=stds[k]
        s.normalize()
        #mask where flux is less than zero
        wv= s.wave.value
        fl= s.flux.value
        fl[fl < 0.0]=np.nan
        #s.toInstrument('WFC3-G141')
        interpstds[k]=interpolate.interp1d(wv, fl, bounds_error=False,fill_value=0.)
    return interpstds


INTERPOLATED_STANDARD_DICT=interpolated_standards()


#used for classification
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
        self._spectral_type=kwargs.get('spt',None)
        self._best_fit_line=None
        self.is_ucd=kwargs.get('is_ucd', False) #flag for UCD candidates

        #load spectrum if given filename 
        
        if 'filepath' in kwargs: self._filepath =kwargs.get('filepath', None)
        if 'filename' in kwargs:  self._filename =kwargs.get('filename', None)
        if 'name' in kwargs: self._filename =kwargs.get('name', None)

        #print (return_path(self._filename))
        if  (self._filename is not None):
            self.filename=self._filename

        if (self._filepath is not None):
        	self.filepath=self._filepath
        	
        if (self._wave is not None) and (self.filepath is None) and (not self.is_ucd):
            self._compute_snr()
            self._splat_spectrum=splat.Spectrum(wave=self._wave, flux=self._flux, noise=self._noise, instrument='WFC3')
            self._best_fit_line=fit_a_line(self)
            ftest=f_test(self)
            for key in  ftest.keys(): 
                setattr(self, key, ftest[key])
            self.normalize()

        if self._spectral_type is not None:
            self.spectral_type=self._spectral_type

        #keep a copy of this object as an attribute
        #read the file locallly for UCDs
        if (self._wave is None ) and self.is_ucd:
            row=(UCD_SPECTRA[UCD_SPECTRA.grism_id==self._filename]).iloc[0]
            self._wave=row.wave
            self._flux=row.flux
            self._noise=row.noise
            self._contam=row.contam
            self._compute_snr()
            self._splat_spectrum=splat.Spectrum(wave=self._wave, flux=self._flux, noise=self._noise, instrument='WFC3')
            self._best_fit_line=fit_a_line(self)
            ftest=f_test(self)
            for key in  ftest.keys(): 
                setattr(self, key, ftest[key])
            self.normalize()

        self.original=copy.deepcopy(self)

    
    def __repr__(self):
        if self._filename is None:return "anon spectrum"
        else: return self._filename


    def reset(self):
        '''
        :Purpose: 
            Restores a Spectrum to its original read-in state, removing scaling and smoothing. 
        :Required Inputs:
            None
        
        :Optional Inputs:
            None
        :Output:
            Spectrum object is restored to original parameters
        '''
        self.__dict__.update(self.original.__dict__)
        self.original = copy.deepcopy(self)
        
    
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
    def spectral_type(self):
        return self._spectral_type

    @spectral_type.setter
    def spectral_type(self, new_type):
        self._spectral_type=new_type
        ftest=f_test(self)
        for key in  ftest.keys():
            setattr(self, key, ftest[key])

    @property
    def index_type(self):
        return splat.classifyByIndex(self.splat_spectrum, ref='allers')


    @property
    def dof(self):
        #convert to use splat dof 
        return self.splat_spectrum.dof
    

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
    	return splat.Spectrum(wave=self._wave, flux=self._flux, noise=self._noise, instrument='WFC3-G141')
        
    @property
    def best_fit_line(self):
        ##save the best fit line as part of the object
        return self._best_fit_line
    
    #@lru_cache(maxsize=128)
    def classify_by_standard(self, **kwargs):
        """
        Uses splat.classifyByStandard to classify spectra using spex standards
        """ 
        self.spectral_type=classify(self, **kwargs)

    def normalize(self, **kwargs):
        """
        :Purpose: Normalize a spectrum to a maximum value of 1 (in its current units)
        :input:
        :ouput:
        :Example:
        """
   

        #rescale the spectrum for lower stuff 
        #should I add a normalized contamination?

        if kwargs.get('rescale_wisp', False):
            up_wave=np.logical_and(self._wave> 1.2, self._wave<1.35)
            down_wave=np.logical_and(self._wave> 0.9, self._wave<1.12)
            scale=np.nanmedian(self._flux[up_wave])/np.nanmedian(self._flux[down_wave])
            self._flux[self._wave< 1.17]=(self._flux[self._wave< 1.17])*scale
        sp=self.splat_spectrum
        sp.normalize(**kwargs)
        self._wave= sp.wave.value
        self._flux=sp.flux.value
        self._noise=sp.noise.value

        #scale the contamination
        if not np.isnan(self._contam).all():
            medflx=np.nanmedian(self._flux[np.logical_and(self.wave>1.2, self.wave<1.6 ) ])
            scl= medflx/np.nanmedian(self._contam[np.logical_and(self.wave>1.2, self.wave<1.6 ) ])
            self._contam=self._contam*scl

        
        return

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

            self._snr= {'snr1':snr1, 'snr2':snr2, 'snr3':snr3, 'snr4':snr4}
        
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

    def add_noise(self, snr, **kwargs):
        """
        add n-sigma noise to the spectrum
        """
        self.normalize()
        snr0=self.snr['snr1']
        self._noise= np.array([x*snr0/snr for x in self.noise])
        self._flux =self.flux+(np.random.normal(np.zeros(len(self.noise)), self.noise))
        self._compute_snr()
        self.classify_by_standard()
        #ftest=f_test(self)
        ns=kwargs.get('nsample', 100)
        if kwargs.get('recompute_indices', False):
            self._indices=measure_indices(self, return_unc=True, nsamples=ns)
        #for key in  ftest.keys():
        #    setattr(self, key, ftest[key])

  

    @property 
    def filepath(self): 
        return self._filepath

    @property
    def f_test(self):
        return self.f
    
    	
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
            survey, stamp_image_path=get_image_path(self._filename, self._filepath)
            self._spectrum_image_path=stamp_image_path
            self._survey=survey
        
        #print (self._filename)
        #sometimes column keys are "col1', 'col2', 'col3', etc.. instead of wave, flux, error
       
        wv= data['wave'] 
        flux=data['flux']
        noise=data['error']
        contam=data['contam']
        self._wave=np.array(wv)/10000.0
        
        self._noise=np.array(noise)
        self._contam=np.array(contam)

        # subtract the contamination as prescribed by wisps people
        self._flux=np.array(flux)-np.array(contam)

        ##apply sensitivity correction from 3d-hst
        if not self._survey == 'wisps':
            with  fits.open(self._spectrum_image_path, memmap=False) as hdu:
                self._sensitivity=hdu[10].data
            #divide by the sensitivity
            self._flux=self._flux/self._sensitivity
            hdu.close()
            del hdu[10].data

        
        #add offset if some of the flux is negative
        #print (self._wave)
        offset_flux=np.nanmin(self._flux[np.where((self._wave >1.4) & (self._wave <1.5))])
        if offset_flux<0.0:
                 self._flux=self._flux+abs(offset_flux)

        self._compute_snr()
        self._indices= measure_indices(self, return_unc=True)
        self.original = copy.deepcopy(self)
        #print (fit_a_line(self))
        self._best_fit_line=fit_a_line(self)
        ftest=f_test(self)
        for key in  ftest.keys():
                setattr(self, key, ftest[key])
        
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
        #re-reading files is inefficient 
        self._filename=new_filename
        if (self._filepath is None ) and (not self.is_ucd):
            survey, spectrum_path, stamp_image_path=parse_path(new_filename, 'v5')
            self._filename=spectrum_path.split('/')[-1]
            self._spectrum_image_path=stamp_image_path
            self._survey=survey
            self.filepath= spectrum_path
        
    @property
    def survey(self):
    	return self._survey
        
    @property
    def spectrum_image(self):
        imgdata=None
        if (self._survey == 'wisps') & (not self.is_ucd):
            with fits.open(self._spectrum_image_path, memmap=False) as imghdu:
                imgdata=imghdu[0].data
            imghdu.close()
            del imghdu
        if (self._survey != 'wisps') & (not self.is_ucd):
            with fits.open(self._spectrum_image_path, memmap=False) as imghdu:
                imgdata=imghdu[5].data-imghdu[8].data
            imghdu.close()
            del imghdu

        if self.is_ucd:
            row=(UCD_SPECTRA[UCD_SPECTRA.grism_id==self._filename]).iloc[0]
            imgdata=row.spectrum_image
        return imgdata


    
    @property
    def sensitivity_curve(self):
    	return self._sensitivity
    
    def plot(self, **kwargs):
        splat_plot.plotSpectrum(self._splat_spectrum, **kwargs)
    
    		


def fit_a_line(spectrum, **kwargs):
    """
    Fit a line, returns a chi-square
    """
    #only fits within the range
    mask=kwargs.get('mask', np.where((spectrum.wave>1.15) & (spectrum.wave <1.65))[0])
    wave=spectrum.wave[mask]
    flux=spectrum.flux[mask]
    noise=spectrum.noise[mask]
    #fit a line from stast linerar regression package
    m, b, r_value, p_value, std_err = stats.linregress(wave, flux)
    line=m*wave+b
    chisqr=np.nansum((flux-line)**2/noise**2)
    #return the line anc chi-square
    return tuple([line, chisqr])


def f_test(spectrum, **kwargs):
    """
    Use an F-test to see wether a line fits better than a spectral standard
    """
    #get the splat spectrum
    s=spectrum.splat_spectrum
    #trim within the same wavelength used to compare to standards
    #normalize
    s.normalize(waverange=[1.15, 1.65])
    s.trim([1.15, 1.65])
    #fit a line
    linefit=fit_a_line(spectrum)
    line= linefit[0]
    linechi=linefit[1]
    if spectrum.spectral_type is None:
        spectrum.classify_by_standard()
    
    #print (spectrum.spectral_type)
    spt=spectrum.spectral_type[0]
    std=STD_DICTS[splat.typeToNum(spt)]

    std.normalize(waverange=[1.2, 1.6])
    spexchi=splat.compareSpectra(s, std,  comprange=[[1.2, 1.6]], statistic='chisqr', scale=True)[0].value

    #calculate f-statistic
    x=spexchi/linechi
    #calculate the f-statistic dfn=2, dfd=1 are areguments
    f=stats.f.cdf(x,  spectrum.dof-1, spectrum.dof-2)
    #return result
    result={'spex_chi':spexchi, 'line_chi':linechi, \
    'x':x, 'f':f, '_best_fit_line': [line, linechi], 'df1':std.dof+spectrum.dof-1, 'df2': spectrum.dof-2  }
    return result

def compute_chi_square(flux, noise, model):
    if (noise==0.0).all(): noise=flux*1e-3
    #mask out less than zero flux
    flux[flux<0.0]=np.nan
    scale=np.nansum((flux*model)/noise**2)/np.nansum(model**2/noise**2)
    return float(np.nansum((flux-scale*model)**2/(noise**2)))

def classify(sp, **kwargs):
    """
    My own classify by standard 
    must be vectorizable and faster than splat

    sp= wisp spectrum object
    """
    #normalize both spectra
    comprange=kwargs.get('comprange', [1.15, 1.65])
    dof=kwargs.get('dof', None)
    mask=None
    #check for multiarray mask
    if len(np.shape(comprange)) <2:
        dof=len(sp.wave[np.logical_and(sp.wave>comprange[0], sp.wave <comprange[1] )])-1
        mask=np.logical_and(sp.wave <= comprange[1], sp.wave >=comprange[0]  )

    else:
        masks=[]
        dof=0.0
        for wv in comprange:
            masks.append(np.logical_and(sp.wave > wv[0], sp.wave < wv[1]))
            dof += len(sp.wave[np.logical_and(sp.wave > wv[0], sp.wave < wv[1])])-1
        mask=np.logical_or.reduce(masks)

    #print (mask, dof)
    #mask
    wave=sp.wave[mask]
    flux=sp.flux[mask]
    noise=sp.noise[mask]
    #dof=kwargs.get('dof', sp.splat_spectrum.dof)

    chisqrs=[]
    for k in splat.STDS_DWARF_SPEX.keys():
        model_f=INTERPOLATED_STANDARD_DICT[k]
        model=model_f(wave)
        #model_noise= model_n(wave)
        #tot_noise= (noise**2 +  model_noise**2)**0.5
        chisqrs.append([compute_chi_square(flux, noise, model), float(make_spt_number(k))])

    #smallest_chi_Square is the clasification
    chisqrs=np.vstack(chisqrs)
    mean, var= np.round(splat.weightedMeanVar(make_spt_number(chisqrs[:,1]), chisqrs[:,0], method='ftest',dof=dof))
    #unc_sys = 0.5
    return mean, (0.5**2+var**2)**0.5    


def classify_by_templates(sp, **kwargs):
    #my hack versions of classifying by templates
    """
    """
    #normalize both spectra
    comprange=kwargs.get('comprange', [1.15, 1.65])
    dof=kwargs.get('dof', None)
    mask=None
    #check for multiarray mask
    if len(np.shape(comprange)) <2:
        dof=len(sp.wave[np.logical_and(sp.wave>comprange[0], sp.wave <comprange[1] )])-1
        mask=np.logical_and(sp.wave <= comprange[1], sp.wave >=comprange[0]  )

    else:
        masks=[]
        dof=0.0
        for wv in comprange:
            masks.append(np.logical_and(sp.wave > wv[0], sp.wave < wv[1]))
            dof += len(sp.wave[np.logical_and(sp.wave > wv[0], sp.wave < wv[1])])-1
        mask=np.logical_or.reduce(masks)

    #print (mask, dof)
    #mask
    wave=sp.wave[mask]
    flux=sp.flux[mask]
    noise=sp.noise[mask]
    #dof=kwargs.get('dof', sp.splat_spectrum.dof)

    #
    tmpls=pd.read_pickle(OUTPUT_FILES+'/validated_templates.pkl')
    chisqrs=[]
    for _, row in tmpls.iterrows():
        model=row.interp(wave)
        #model_noise= model_n(wave)
        #tot_noise= (noise**2 +  model_noise**2)**0.5
        chisqrs.append([compute_chi_square(flux, noise, model), float(row.spt)])
     #smallest_chi_Square is the clasification
    chisqrs=np.vstack(chisqrs)

    mean, var= np.round(splat.weightedMeanVar(make_spt_number(chisqrs[:,1]), chisqrs[:,0], method='ftest',dof=dof))
   
    return mean, (0.5**2+var**2.0)**0.5    





def distance(mags, spt, spt_unc):
    """
    mags is a dictionary of bright and faint mags

    set a bias 
    """
    res={}
    
    f110w=mags['F110W']
    f140w=mags['F140W']
    f160w=mags['F160W']

    relations=POLYNOMIAL_RELATIONS['abs_mags']
    nsample=1000

    for k in mags.keys():
        #take the standard deviation
        spt=make_spt_number(spt)

        absmag_scatter=relations[k][1]
        spts=np.random.normal(spt, spt_unc, nsample)
        #trim out spectral types outside range of validitiy
        mask=(spts<15)  & (spts >40)
        absmags=(relations[k][0])(spts)[~mask] 

        #total_uncertainty
        mag_unc=(absmag_scatter**2+mags[k][1]**2)**0.5
        relmags=np.random.normal(mags[k][0], mag_unc, nsample)[~mask]
        dists=get_distance(absmags, relmags)
        
        #res[str('dist')+k]=np.nanmedian(dists)
        #res[str('dist_er')+k]=np.nanstd(dists)
        res['dist'+k]=dists

    return res
    
def kde_statsmodels_m(x, x_grid, **kwargs):
    """
    multivariate kde
    """
    model=kde.KDEMultivariate(x, bw='normal_reference', var_type='c')
    return model.cdf(x_grid)

if __name__=='__main__':
	pass
    