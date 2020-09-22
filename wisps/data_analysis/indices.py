#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains all methods for manulating spectral indices 
borrowed from a previous version of splat (github.com/aburgasser/splat)
"""

__author__= 'caganze'

from .initialize import * 

#def meaure_indices()


from scipy.integrate import trapz        # for numerical integration
from scipy.interpolate import interp1d   #for 1-d interpolation
import splat


#change indices later to 1.50

def measure_all_indices(wave, flux, noise):

    wavranges=[[[1.15, 1.20], [1.246, 1.295]],
    [[1.38, 1.43],  [1.15, 1.20]],  
    [[1.56, 1.61],  [1.15, 1.20]],  
    [[1.62,1.67],   [1.15, 1.20]],  
    [[1.38, 1.43],  [1.246, 1.295]],
    [[1.56, 1.61],  [1.246, 1.295]],
    [[1.62,1.67],   [1.246, 1.295]],
    [[1.56, 1.61],  [1.38, 1.43]],
    [[1.62,1.67],   [1.38, 1.43]],
    [[1.62,1.67],   [1.56, 1.61]],
    [[1.38, 1.43],  [1.15, 1.20],   [1.246, 1.295]],
    [[1.38, 1.43],  [1.15, 1.20],   [1.56, 1.61]],
    [[1.15, 1.20], [1.62,1.67],  [1.246, 1.295]],
    [[1.15, 1.20], [1.62,1.67],   [1.56, 1.61]],
    [[1.38, 1.43], [1.62,1.67], [1.246, 1.295]],
    [[1.38, 1.43], [1.62,1.67],   [1.56, 1.61]]]
    
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


def measure_indices(sp,**kwargs):
    """
    sp must be a Spectrum object
    roughly similar to splat.measureIndices (github.com/aburgasser/splat) 
    """
    sp.normalize()
    return  measure_all_indices(sp.wave, sp.flux, sp.noise)


