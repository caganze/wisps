#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
#Selection function routines
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

        for i in tqdm(np.arange(nsample)):
            spectr.normalize(waverange=[1.25,1.5])
            mu= np.nanmedian(spectr.noise[mask])
            sigma=np.nanstd(spectr.noise[mask])
            noise=5.0*np.random.lognormal(mu,sigma,len(spectr.flux))
            spectr.add_noise(noise=noise)
            n1=np.nanmedian(spectr.original.snr['snr1'])
            n2=np.nanmedian(spectr.snr['snr1'])
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
      
        return df

    def make_data(self, spectra, spts,  **kwargs):
        """
        create a selection function from a list of spectra and spts
        """
        results=[]
        for sx, spt in zip(spectra, spts):
            results.append(self.generate_spectra(sx, spt, **kwargs))

        results=np.array(results)
        print (len(results), len(results[0]))
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


def create_selection_function(**kwargs):
    """
    Create a selection with data 
    """
    #optional inputs
    output_file=kwargs.get('output_file', wisps.OUTPUT_FILES+'/selection_function.pkl')
    spectra_file=kwargs.get('spectra_file', wisps.OUTPUT_FILES+'/l_t_dwarfs.pkl')
    nsamples=kwargs.get('nsamples', 10)

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
    sf.make_data(spectra, spts, nsample=10)

    with open(output_file, 'wb') as file:
        pickle.dump(sf,file)

    return sf







