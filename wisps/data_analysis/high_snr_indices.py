#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
#Re-measure indices for high SNR objects only
########################



import wisps
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial
import numba 


def get_object(filename):
    print (filename)
    sp=wisps.Spectrum(name=filename)
    f_test={"f_test": sp.f_test, 'line_chi': sp.line_chi, 'spex_chi': sp.spex_chi, 'spt_new': sp.spectral_type, 'spt': sp.spectral_type}
    res_dict= {**sp.snr, **sp.indices, **f_test}
    return res_dict



def make_data(spectranames, **kwargs):
    """
    measure parameters for a bunch of spectra
    """
    results=[]
    iterables=([spectranames])
    method=partial(get_object)
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    return pd.DataFrame.from_records(results)

def run_objects():
 	#important=
	pred_df=(wisps.datasets['stars'])
	pred_df=pred_df[pred_df.snr2>3.]
	d=make_data(pred_df['grism_id'])

	d.to_pickle(wisps.OUTPUT_FILES+'/highsnr_obejcts.pkl')

if __name__=='__main__':
    run_objects()

