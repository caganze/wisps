#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
#Selection function routines

#NEED MORE COMMENTS
########################

#imports
import splat
import wisps
import matplotlib.pyplot as plt
from wisps import datasets,make_spt_number
from wisps.simulations import selection_function
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import splat.empirical as spem
import copy
from wisps.data_analysis import selection_criteria as sel_crt
import pickle
import numba
import copy


from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial


def add_noise_to_spectrum(sp, snr):
    #if I propose a larger SNR don't do anything to save time
    sp.reset()
    sp_old=sp.spectral_type
    sp.add_noise(snr, nsample=1)
    f_test={"f_test": sp.f_test, 'line_chi': sp.line_chi, 'spex_chi': sp.spex_chi, 'spt_new': sp.spectral_type, 'sp_old': sp_old}
    res_dict= {**sp.snr, **fast_measure_indices(sp), **f_test}
    sp.reset()
    return res_dict
    
def add_multiple_noises(sp, noises):
    res=list(map(lambda x: add_noise_to_spectrum(sp, x), noises))
    return res


def fast_measure_indices(sp):
    #fast wway to measure indices without monte-carlo sampling or interpolation
    regions=np.array([[[1.15, 1.20], [1.246, 1.295]],
         [[1.38, 1.43],  [1.15, 1.20]], 
         [[1.56, 1.61],  [1.15, 1.20]], 
         [[1.62,1.67],   [1.15, 1.20]], 
        [[1.38, 1.43],  [1.246, 1.295]], 
         [[1.56, 1.61],  [1.246, 1.295]],
         [[1.62,1.67],   [1.246, 1.295]], 
         [[1.56, 1.61],  [1.38, 1.43]],
         [[1.62,1.67],   [1.38, 1.43]],
         [[1.62,1.67],   [1.56, 1.61]]])
    labels=wisps.INDEX_NAMES
    res=pd.Series()
    res.columns=labels
    #loop over ratios 
    for r, l in zip(regions, labels):
        flx1=np.nanmedian(sp.flux[np.where((sp.wave>r[0][0]) & (sp.wave<r[0][1]))[0]])
        flx2=np.nanmedian(sp.flux[np.where((sp.wave>r[1][0]) & (sp.wave<r[1][1]))[0]])
        res[l]= (flx1/flx2, 0.0)
    return dict(res)


def make_data(spectra, **kwargs):
    """
    create a selection function from a list of spectra and spts
    """
    results=[]
    nsample=kwargs.get("nsample", 1000)
    #for sx, spt in zip(spectra, spts):
     #   results.append(self.generate_spectra(sx, spt, **kwargs))
    #run this in parallel

    snrs=10**np.random.uniform(-1,3,(len(spectra), nsample))
    iterables=([spectra, snrs])

    method=partial(add_multiple_noises)
    with ThreadPoolExecutor(max_workers=1000) as executor:
        results=list(tqdm(executor.map( method, *iterables, timeout=None, chunksize=100)))

    #results=[x for x in futures]

    return pd.concat(pd.DataFrame.from_records(results))

def create_selection_function(**kwargs):
    """
    Create a selection with data 
    """
    #optional inputs
    output_file=kwargs.get('output_file', wisps.OUTPUT_FILES+'/selection_function.pkl')
    spectra=pd.read_pickle(wisps.OUTPUT_FILES+'/l_t_dwarfs_spex.pkl')
    
    splat.initializeStandards()
    
    #set up the selection 
    def convert_to_string(x):
        if isinstance(x, str):
            return x
        else:
            return splat.typeToNum(x)

    res=make_data(spectra, **kwargs)

    with open(output_file, 'wb') as file:
        pickle.dump(res,file)

    return 