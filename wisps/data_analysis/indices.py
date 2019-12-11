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


def measure_indices(s,**kwargs):
    """
    sp must be a Spectrum object
    roughly similar to splat.measureIndices (github.com/aburgasser/splat) 
    """
    sp=s.splat_spectrum
    sample_type=kwargs.get("sample","median")
    ns=kwargs.get("nsamples", 100)
    #names = ['index-1','index-2', 'index-3', 'index-4', 'index-5', 'index-6','index-7', 'index-8', 'index-9', 'index-10' ]
    names=INDEX_NAMES
    inds = np.zeros(len(names))
    errs = np.zeros(len(names))
    inds[0], errs[0]= __measure_index(sp, [1.15, 1.20], [1.246, 1.295], method='ratio',sample=sample_type,nsamples=ns)
    inds[1], errs[1]= __measure_index(sp, [1.38, 1.43],  [1.15, 1.20],  method='ratio',sample=sample_type,nsamples=ns)
    inds[2], errs[2]= __measure_index(sp, [1.56, 1.61],  [1.15, 1.20],  method='ratio',sample=sample_type,nsamples=ns)
    inds[3], errs[3]= __measure_index(sp, [1.62,1.67],   [1.15, 1.20],   method='ratio',sample=sample_type,nsamples=ns)
    inds[4], errs[4]= __measure_index(sp, [1.38, 1.43],  [1.246, 1.295],method='ratio',sample=sample_type,nsamples=ns)
    inds[5], errs[5]= __measure_index(sp, [1.56, 1.61],  [1.246, 1.295],method='ratio',sample=sample_type,nsamples=ns)
    inds[6], errs[6]= __measure_index(sp, [1.62,1.67],   [1.246, 1.295],method='ratio',sample=sample_type,nsamples=ns)
    inds[7], errs[7]= __measure_index(sp, [1.56, 1.61],  [1.38, 1.43],method='ratio',sample=sample_type,nsamples=ns)
    inds[8], errs[8]= __measure_index(sp, [1.62,1.67],   [1.38, 1.43],method='ratio',sample=sample_type,nsamples=ns)
    inds[9], errs[9]= __measure_index(sp, [1.62,1.67],   [1.56, 1.61],method='ratio',sample=sample_type,nsamples=ns)

    if kwargs.get('return_unc', False):
        result = {names[i]: (inds[i],errs[i]) for i in np.arange(len(names))}
    if not kwargs.get('return_unc', False):
        result = {names[i]: inds[i] for i in np.arange(len(names))}
    return result

def fast_measure_indices(sp, regions, labels, **kwargs):
    #fast wway to measure indices without monte-carlo sampling or interpolation
    res=pd.Series()
    res.columns=labels
    #loop over ratios 
    for r, l in zip(regions, labels):
        flx1=sp.flux(np.where((sp.wave>r[0][0]) & (sp.wave<r[0][1]))[0])
        flx2=sp.flux(np.where((sp.wave>r[1][0]) & (sp.wave<r[1][1]))[0])
        res[l]= flx1/flx2
    return dict(res)

def __measure_index(sp,*args,**kwargs):
    """
    internal function for index measurements
    from splat
    
    """
    # keyword parameters
    method = kwargs.get('method','ratio')
    sample = kwargs.get('sample','integrate')
    nsamples = kwargs.get('nsamples',1000)
    noiseFlag = kwargs.get('nonoise',False)
    # create interpolation functions
    w = np.where(np.isnan(sp.flux) == False)
    f = interp1d(sp.wave.value[w],sp.flux.value[w],bounds_error=False,fill_value=0.)
    w = np.where(np.isnan(sp.noise) == False)
    # note that units are stripped out
    if (np.size(w) != 0):
        s = interp1d(sp.wave.value[w],sp.noise.value[w],bounds_error=False,fill_value=np.nan)
        noiseFlag = False
    else:
        s = interp1d(sp.wave.value[:],sp.noise.value[:],bounds_error=False,fill_value=np.nan)
        noiseFlag = True
    # error checking on number of arguments provided
    if (len(args) < 2):
        print('measureIndex needs at least two samples to function')
        return np.nan, np.nan
    elif (len(args) < 3 and (method == 'line' or method == 'allers' or method == 'inverse_line')):
        print(method+' requires at least 3 sample regions')
        return np.nan, np.nan
    # define the sample vectors
    value = np.zeros(len(args))
    value_sim = np.zeros((len(args),nsamples))
    # loop over all sampling regions
    for i,waveRng in enumerate(args):
        xNum = (np.arange(0,nsamples+1.0)/nsamples)* \
            (np.nanmax(waveRng)-np.nanmin(waveRng))+np.nanmin(waveRng)
        yNum = f(xNum)
        yNum_e = s(xNum)
        #first compute the actual value
        if (sample == 'integrate'):
            value[i] = trapz(yNum,xNum)
        elif (sample == 'average'):
            value[i] = np.nanmean(yNum)
        elif (sample == 'median'):
            value[i] = np.nanmedian(yNum)
        elif (sample == 'maximum'):
            value[i] = np.nanmax(yNum)
        elif (sample == 'minimum'):
            value[i] = np.nanmin(yNum)
        else:
            value[i] = np.nanmean(yNum)
        # now do MonteCarlo measurement of value and uncertainty
        for j in np.arange(0,nsamples):
            # sample variance
            if (np.isnan(yNum_e[0]) == False):
                yVar = yNum+np.random.normal(0.,1.)*yNum_e
            # NOTE: I'M NOT COMFORTABLE WITH ABOVE LINE - SEEMS TO BE TOO COARSE OF UNCERTAINTY
            # BUT FOLLOWING LINES GIVE UNCERTAINTIES THAT ARE WAY TOO SMALL
            #                yVar = numpy.random.normal(yNum,yNum_e)
            #                yVar = yNum+numpy.random.normal(0.,1.,len(yNum))*yNum_e
            else:
                yVar = yNum
            # choose function for measuring indices
            if (sample == 'integrate'):
                value_sim[i,j] = trapz(yVar,xNum)
            elif (sample == 'average'):
                value_sim[i,j] = np.nanmean(yVar)
            elif (sample == 'median'):
                value_sim[i,j] = np.nanmedian(yVar)
            elif (sample == 'maximum'):
                value_sim[i,j] = np.nanmax(yVar)
            elif (sample == 'minimum'):
                value_sim[i,j] = np.nanmin(yVar)
            else:
                value_sim[i,j] = np.nanmean(yVar)
    # compute index based on defined method
    # default is a simple ratio
    if (method == 'ratio'):
        val = value[0]/value[1]
        vals = value_sim[0,:]/value_sim[1,:]
    elif (method == 'line'):
        val = 0.5*(value[0]+value[1])/value[2]
        vals = 0.5*(value_sim[0,:]+value_sim[1,:])/value_sim[2,:]
    elif (method == 'inverse_line'):
        val = 2.*value[0]/(value[1]+value[2])
        vals = 2.*value_sim[0,:]/(value_sim[1,:]+value_sim[2,:])
    elif (method == 'change'):
        val = 2.*(value[0]-value[1])/(value[0]+value[1])
        vals = 2.*(value_sim[0,:]-value_sim[1,:])/(value_sim[0,:]+value_sim[1,:])
    elif (method == 'allers'):
        val = (((np.nanmean(args[0])-np.nanmean(args[1]))/(np.nanmean(args[2])-np.nanmean(args[1])))*value[2] \
            + ((np.nanmean(args[2])-np.nanmean(args[0]))/(np.nanmean(args[2])-np.nanmean(args[1])))*value[1]) \
            /value[0]
        vals = (((np.nanmean(args[0])-np.nanmean(args[1]))/(np.nanmean(args[2])-np.nanmean(args[1])))*value_sim[2,:] \
            + ((np.nanmean(args[2])-np.nanmean(args[0]))/(np.nanmean(args[2])-np.nanmean(args[1])))*value_sim[1,:]) \
            /value_sim[0,:]
    else:
        val = value[0]/value[1]
        vals = value_sim[0,:]/value_sim[1,:]

    # output mean, standard deviation
    #print (vals)
    if (noiseFlag):
        return val, np.nan
    else:
        return val, np.nanstd(vals)