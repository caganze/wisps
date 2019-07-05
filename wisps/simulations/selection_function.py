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





class SF():

    def __init__(self, **kwargs):
        #initialization
        self._weight_function=None
        self.crts=None #selection criteria
        self.f_cut=None #f_test cut
        self.snr_cut=None #snr cut
        self.data=None

    @property
    def weight_function(self):
        """
        The weight function of each of a spectral type
        """
        return self._weight_function

    def generate_spectra(self, spectr, spt, nsample=10):
        """
        generate a bunch of spectra  at diffeent SNR levels
        sp: a wisps Source object i.e it must have photometry
        spt: spectral type
        nsample: number of spectra
        """
        @np.vectorize
        @numba.jit
        def return_noised_spectrum(snr):
            ##returns a specrtum parameters 
            #after applying noise 
            spectr.normalize(wave_range=[1.1, 1.65])
            spectr.add_noise(snr)
            vals=(spectr, fast_measure_indices(spectr),  {'f':spectr.f_test, 'spt':spectr.spt, 
                'line_chi':spectr.line_chi, 'spex_chi':spectr.spex_chi},  spectr.snr)
            return  vals
        
        tg_snrs=np.logspace(3.0, 0., nsample)
        spectra, indices, ftest, snrs=return_noised_spectrum(tg_snrs)
        spts=[spt for x in spectra]
        return spectra, spts,  pd.DataFrame.from_records(indices), pd.DataFrame.from_records(snrs), pd.DataFrame.from_records(ftest)

    def select(self, df, to_use, **kwargs):
        """
        Use selection criteria on a sample
        df: the sample
        to_use: a dictionary with criteria name and 
        """
        cands=[]
        snr=self.snr_cut
        f_test=self.f_cut
        for k in to_use.keys():
            crt=self.crts[k]
            cands.append(crt.select(wisps.Annotator.reformat_table(df), shapes_to_use=to_use[k]))

        flatten = lambda l: np.array([i for sublist in l for item in sublist for i in item])
        ncands=np.unique(flatten(cands))

        df['selected']=0
        df.selected[(df.snr1>snr) & (df.f> f_test) & (df.Names.isin(ncands))]=1
        #df.selected[(df.snr1>snr) & (df.f> f_test)  &  ]=1
      
      
        self.data=df
    
    @numba.jit(parallel=True, cache=True)
    def probability_of_selection( self, spt, snr):
        """
        probablity of selection for a given snr and spt
        """
        #self.data['spt']=self.data.spt.apply(splat.typeToNum)
        df=self.data
        floor=np.floor(spt)
        floor2=np.log10(np.floor(snr))
        return np.nanmean(df.selected[(df.spt==floor) &(df.snr1.apply(np.log10).between(floor2, floor2+.3))])

    def make_data(self, spectra, spts,  **kwargs):
        """
        create a selection function from a list of spectra and spts
        """
        results=[]
        #for sx, spt in zip(spectra, spts):
         #   results.append(self.generate_spectra(sx, spt, **kwargs))
        #run this in parallel
        iterables=([spectra, spts ])
        print (iterables)
        method=partial(self.generate_spectra, nsample=kwargs.get('nsample', 10))
        with ThreadPoolExecutor(max_workers=1000) as executor:
            futures=executor.map( method, *iterables, timeout=None, chunksize=100)

        results=np.array([x for x in futures])

        df1=pd.concat(results[:,-2]).reset_index(drop=True)
        df2=pd.concat(results[:,-1]).reset_index(drop=True)
        df3=pd.concat(results[:,-3]).reset_index(drop=True)

        df=df1.join(df2).join(df3)
        df=df.drop('spt', axis=1)
        df['spt']=np.concatenate(results[:,1])
        df['spectra']=np.concatenate(results[:,0])
        df['Names']=np.array(['spec'+str(x) for x in range(0, len(df))])

        #add data to object
        self.data=df

        #add data to object
        df['spt']=df.spt.apply(make_spt_number)
        self.data=df

        print (self.data)

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
    return pd.Series(dict(res))


def create_selection_function(**kwargs):
    """
    Create a selection with data 
    """
    #optional inputs
    output_file=kwargs.get('output_file', wisps.OUTPUT_FILES+'/selection_function.pkl')
    spectra_file=kwargs.get('spectra_file', wisps.OUTPUT_FILES+'/l_t_dwarfs_spex.pkl')
    nsample=kwargs.get('nsample', 10)

    splat.initializeStandards()
    #paramaters from selection function I used
    crts=sel_crt.crts_from_file()
    to_use={'H_2O-2/J-Cont CH_4/H-Cont':['L0-L5'],
        'H_2O-1/J-Cont CH_4/H_2O-2':['L5-T0'],
        'H-cont/H_2O-1 H_2O-2/J-Cont':['M7-L0'],
        'H_2O-1/J-Cont H_2O-2/H_2O-1':['T0-T5'],
        'H_2O-1/J-Cont CH_4/H-Cont':['Y dwarfs', 'T5-T9'],
        'CH_4/H_2O-1 H-cont/J-Cont':['subdwarfs']}
    sf=SF()
    #set up the selection 
    def convert_to_string(x):
        if isinstance(x, str):
            return x
        else:
            return splat.typeToNum(x)

    #f-test and snr cut
    sf.crts=crts
    sf.f_cut=0.5
    sf.snr_cut=3.0
    #save it in the 
    spectra=pd.read_pickle(spectra_file)
    spts=np.array([convert_to_string(s.spt) for s in spectra])
    sf.make_data(spectra, spts, **kwargs)
    sf.select(sf.data, to_use)

    with open(output_file, 'wb') as file:
        pickle.dump(sf,file)

    return sf