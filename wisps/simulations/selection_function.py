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
from wisps import datasets
from wisps.simulations import selection_function
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import splat.empirical as spem
import copy
from wisps.data_analysis import selection_criteria as sel_crt
import pickle

import copy

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

    def generate_spectra(self, spectr, spt, nsample=5):
        """
        generate a bunch of spectra  at diffeent SNR levels
        sp: a wisps Source object i.e it must have photometry
        spt: spectral type
        nsample: number of spectra
        """
        mag=float(spectr.mags['F160W'][0])
        ratios=[]
        snrs=[]
        indices=[]
        f_test=[]
        spts=[]
        spectra=[]
        #degrade the object
        #create a mask in the region of interest
        mask=np.where((spectr.wave>1.1) & (spectr.wave<1.7))[0]

        spectr.normalize(waverange=[1.25,1.65])

        spectra.append(spectr)
        snrs.append(spectr.snr)
        indices.append(fast_measure_indices(spectr))
        f_test.append(spectr.f_test())
        spts.append(spt)
        ratios.append(1.0)

        for i in tqdm(np.arange(nsample)):
            n1=spectr.original.snr['snr1']
            #normalize the spectrum between 1.25 and 1.65 microns
            spectr.normalize(waverange=[1.25,1.65])
            #find the median flux then add lognormal noise
            mu= np.nanmedian(spectr.noise[mask])
            #fin the std
            sigma=np.nanstd(spectr.noise[mask])
            #add 1-sigma log-normal noise
            noise=0.001*n1*np.random.random()*np.random.lognormal(mu,sigma,len(spectr.flux))
            spectr.add_noise(noise=noise)
            #calculate the ratio in snr
            
            n2=spectr.snr['snr1']
            #append these data
            ratios.append(n2/n1)
            snrs.append(pd.Series(spectr.snr))
            indices.append(pd.Series(spectr.indices))
            f_test.append(spectr.f_test())
            spts.append(spt)
            spectra.append(spectr)
            #reset the object after degradation
            spectr.reset()
        
        #calculate distances
        absmag=spem.typeToMag(spt,'NICMOS F160W',set='dupuy')[0]
        #colors
        colors=-2.5*np.log10(ratios)
        d0=float(spectr.distance['val'].value)
        d=d0+np.array(10**(colors/5.+1.))
    
        return spectra, spts, d, pd.DataFrame(indices), pd.DataFrame(snrs), pd.DataFrame(f_test)

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
      
        self.data=df

    def make_data(self, spectra, spts,  **kwargs):
        """
        create a selection function from a list of spectra and spts
        """
        results=[]
        for sx, spt in zip(spectra, spts):
            results.append(self.generate_spectra(sx, spt, **kwargs))

        results=np.array(results)
        df1=pd.concat(results[:,3]).reset_index(drop=True)
        df2=pd.concat(results[:,-1]).reset_index(drop=True)
        df3=pd.concat(results[:,4]).reset_index(drop=True)
        df=df1.join(df2).join(df3)
        df=df.drop('spt', axis=1)
        df['dist']=np.concatenate(results[:,2])
        df['spt']=np.concatenate(results[:,1])
        df['spectra']=np.concatenate(results[:,0])
        df['Names']=np.array(['spec'+str(x) for x in range(0, len(df))])

        #add data to object
        self.data=df

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
        res[l]= flx1/flx2
    return dict(res)

def create_selection_function(**kwargs):
    """
    Create a selection with data 
    """
    #optional inputs
    output_file=kwargs.get('output_file', wisps.OUTPUT_FILES+'/selection_function.pkl')
    spectra_file=kwargs.get('spectra_file', wisps.OUTPUT_FILES+'/l_t_dwarfs.pkl')
    nsample=kwargs.get('nsample', 10)

    splat.initializeStandards()
    #paramaters from selection function I used
    crts=sel_crt.crts_from_file()
    to_use={'H_2O-2/H_2O-1 CH_4/J-Cont':['subdwarf'],
            'CH_4/H_2O-1 H-cont/J-Cont':['T0-T5'],
            'H_2O-1/J-Cont H-cont/J-Cont':['T5-T9'],
            'H-cont/H_2O-1 H_2O-2/J-Cont':['M7-L0'],
            'H_2O-2/H_2O-1 CH_4/H-Cont':['L5-T0'],
            'H_2O-2/J-Cont H-cont/J-Cont':['L0-L5']}
    #create a selection function
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
    spts=np.array([convert_to_string(s.spectral_type) for s in spectra])
    sf.make_data(spectra, spts, **kwargs)
    sf.select(sf.data, to_use)

    with open(output_file, 'wb') as file:
        pickle.dump(sf,file)

    return sf







