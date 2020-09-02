##############################################################################

################## self-containing code measuring indices on bridges ##########

#imports
import numpy as np 
import pandas as pd
from astropy.io import ascii

#constants

INDEX_NAMES=['H_2O-1/J-Cont', 'H_2O-2/H_2O-1', 'H-cont/H_2O-1', 'CH_4/H_2O-1',
       'H_2O-2/J-Cont', 'H-cont/J-Cont', 'CH_4/J-Cont', 'H-cont/H_2O-2',
       'CH_4/H_2O-2', 'CH_4/H-Cont', 'H_2O-1+H_2O-2/J-Cont',
       'H_2O-1+H_2O-2/H-Cont', 'H_2O-1+CH_4/J-Cont', 'H_2O-2+CH_4/J-Cont',
       'H_2O-1+CH_4/H-Cont', 'H_2O-2+CH_4/H-Cont']

def read_file_into_df(filepath):
	# read a file into flux, noise,wave
	#returns a numpy array
	df=ascii.read(filepath).to_pandas()
	return df.values


def measure_all_indices(wave, flux, noise):
    wavranges=[[[1.15, 1.20], [1.246, 1.295]],
    [[1.38, 1.43],  [1.15, 1.20]],  
    [[1.55, 1.60],  [1.15, 1.20]],  
    [[1.62,1.67],   [1.15, 1.20]],  
    [[1.38, 1.43],  [1.246, 1.295]],
    [[1.55, 1.60],  [1.246, 1.295]],
    [[1.62,1.67],   [1.246, 1.295]],
    [[1.55, 1.60],  [1.38, 1.43]],
    [[1.62,1.67],   [1.38, 1.43]],
    [[1.62,1.67],   [1.55, 1.60]],
    [[1.38, 1.43],  [1.15, 1.20],   [1.246, 1.295]],
    [[1.38, 1.43],  [1.15, 1.20],   [1.55, 1.60]],
    [[1.15, 1.20], [1.62,1.67],  [1.246, 1.295]],
    [[1.15, 1.20], [1.62,1.67],   [1.55, 1.60]],
    [[1.38, 1.43], [1.62,1.67], [1.246, 1.295]],
    [[1.38, 1.43], [1.62,1.67],   [1.55, 1.60]]]
    return dict(zip(INDEX_NAMES, [measure_median_index(wave, flux, noise, x) for  x in wavranges]))
    
    
def measure_median_index(wave, flux, noise, *args):
	#return flux ratios between two wavelenght regions
	#waverange should be a list of wavelenght ranges
    #the numerator first then the denominator
    #if len(waverange) ==3, add the first two
	#waverange1 should be the denominator
	#select the region of the spectrum 
    
    #add a buffer above flux level o avoid nans 
    #intialize mask arrays
    vals=np.zeros((len(*args), 1000))
    for idx, wv in enumerate(*args):
        mask=np.logical_and(wave > wv[0], wave < wv[1])
        #propagate noise measurement
        vals[idx]=np.random.normal(np.nanmedian(flux[mask]), np.nanmedian(abs(noise[mask])), 1000)
        
    #for two
    if len(vals)==2:
        return np.nanmedian(vals[0]/vals[1]), np.nanstd(vals[0]/vals[1])
    
    #for three
    elif len(vals)==3:
        return np.nanmedian((vals[0]+vals[1])/vals[-1]), np.nanstd((vals[0]+vals[1])/vals[-1])
    
    else:
        return (np.nan, np.nan)