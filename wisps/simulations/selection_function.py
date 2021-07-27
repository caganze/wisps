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
    sp.add_noise(snr, nsample=1, recompute_indices= True)
    f_test={"f_test": sp.f_test, 'line_chi': sp.line_chi, 'spex_chi': sp.spex_chi, 'spt_new': sp.spectral_type, 'sp_old': sp_old, 'dof': sp.dof}
    res_dict= {**sp.snr, **sp.indices, **f_test}
    sp.reset()
    return res_dict
    
def add_multiple_noises(sp, noises):
    res=list(map(lambda x: add_noise_to_spectrum(sp, x), noises))
    return res



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
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=np.array([x for x in futures]).flatten()

    return pd.DataFrame.from_records(results)

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