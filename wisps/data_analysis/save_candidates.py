#!/usr/bin/python
# -*- coding: utf-8 -*-

from .initialize import *
import pandas as pd
import numpy as np
import glob
from .photometry import Source, get_multiple_sources
from .spectrum_tools import Spectrum
from ..utils.tools import get_distance, make_spt_number
from tqdm import tqdm
import numba
import pickle

cands = pd.read_pickle(OUTPUT_FILES + '/selected_by_indices.pkl')
import os
SPECTRA_PATH = os.path.dirname(WISP_PATH.split('wisps')[0] + 'wisps'
                               + '//spectra//')


# a lot of routines to make my visual inspection of candidates a feedback loop

def plot_name(name):
    fname = SPECTRA_PATH + '/indices/' + name.replace('-', '_') \
        + '.jpeg'

    # plot(name, fname)

    if os.path.isfile(fname):
        pass
    else:
        plot(name, fname)


def get_cand_grism_ids():

    @numba.jit
    def format_name(name):
        n = name.split('/')[-1]
        n = n.split('.jpeg')[0]
        if n.startswith('par'):
            n = n.replace('_', '-', 1)
        if not n.startswith('par'):
            n = n.replace('_', '-', 2)
        return n
    cands = glob.glob(SPECTRA_PATH + '/indices/*')
    lcands = [format_name(x) for x in cands]

    # save this into the new candidates files
    # df=....COMBINED_PHOTO_SPECTRO_DATA
    # print ((df[df.grism_id.isin(lcands)]).grism_id.values)
    # df[df.grism_id.isin(lcands)].to_pickle(LIBRARIES+'/candidates.pkl')

    df = pd.DataFrame()
    df['grism_id'] = lcands
    df.to_pickle(LIBRARIES + '/candidates_ids.pkl')
    return lcands


def save_cands():
    cands.grism_id.apply(plot_name)


def save_again():

    # save again in the same file

    get_cand_grism_ids()
    df = pd.read_pickle(LIBRARIES + '/candidates.pkl')
    df.grism_id.apply(plot_name)


def look_at_all():
    import wisps

    # df=pd.read_csv(wisps.LIBRARIES+'/selected_by_neural.csv')

    dfs = []
    for f in ['/subdarfs_nn_preds.csv', '/subdarfs_rf_preds.csv',
              '/ydarfs_nn_preds.csv', '/ydarfs_rf_preds.csv']:
        dfs.append(pd.read_csv(wisps.LIBRARIES + f))
    df = pd.concat(dfs)

    # df=df[df.snr1>2.]
    # print (df)

    # remove files where file names exist

    

    # df['grism_id']=df.grism_id.apply(lambda x : x.replace('g141', 'G14

    #sources = get_multiple_sources(df.grism_id.values)  # [~exist_flag])

    # df.to_pickle(LIBRARIES+'/sd_candidates_spectra.pkl')
    output_file=LIBRARIES + '/sd_candidates_spectra.pkl'
    #with open(output_file, 'wb') as file:
    #pickle.dump(sources, file)
    srcs = pd.read_pickle(output_file)
    srcs=[s for s in srcs if s is not None]
    fnames = np.array([SPECTRA_PATH + '/sdcandidates/'
                      + s.name.lower().replace('-', '_') + '.jpeg' for s in
                      srcs])

    exist_flag = np.array([os.path.isfile(fname) for fname in fnames])

    for s in tqdm(np.array(srcs)[~exist_flag]):
        if s is not None:
            try:
                fname = SPECTRA_PATH + '/sdcandidates/' + s.name.replace('-'
                        , '_') + '.jpeg'
                s.plot(save=True, filename=fname.strip(),
                       compare_to_sds=True)
            except:
                fname = SPECTRA_PATH + '/sdcandidates/' + s.name.replace('-', '_') + '.jpeg'
                s=wisps.Source(name=s.name.lower())
                s.plot(save=True, filename=fname.strip(),
                           compare_to_sds=True)