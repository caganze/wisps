#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: caganze

purpose: to fit every spectrum to a line and to standards, get chi-square and f-test values

"""
from  .initialize import *
from .spectrum_tools import Spectrum

from astropy.io import fits, ascii
from astropy.table import Table
import numpy as np 
import pandas as pd
from scipy import interpolate, stats
import splat

def f_test(x):
    """
    f-test statistic with defualt degrees of freedom
    """
    return stats.f.pdf(x, 2, 1, 0, scale=1)

def fit_a_line(wave, flux, noise):
    """
    Fit a line, returns a chi-square
    """
    m, b, r_value, p_value, std_err = stats.linregress(wave,flux)
    line=m*wave+b
    chisqr=np.nansum((flux-line)**2/noise**2)
    #print (chisqr)
    return line, chisqr

def compare_to_both(grism_id):
    """
    compare chi-square from line vs chi-square from standard
    """
    print (grism_id)
    if grism_id.startswith('Par'):
        result= pd.Series({'spex_chi':np.nan, 'line_chi':np.nan, 'spt': np.nan})
    else:
    	sp=Spectrum(name=grism_id)
    	s=sp.splat_spectrum
    	s.trim([1.15, 1.65])
    	line, chi=fit_a_line(s.wave.value, s.flux.value, s.noise.value)
    	spt, spexchi=splat.classifyByStandard(s, return_statistic=True, fit_ranges=[[1.15, 1.65]], plot=False)
    	result=pd.Series({'spex_chi':spexchi, 'line_chi':chi, 'spt': spt})
    	#calculate the f-statistic
    return result


   
def combined_wisp_hst_catalogs():
    """
    combine both hst-3d and wisps into one big file with all the information
    """
    
    #read in the photometry
    hst3d_phot=pd.read_csv(OUTPUT_FILES+'/hst3d_photometry_all.csv')
    wisp_phot=pd.read_csv(OUTPUT_FILES+'/wisp_photometry.csv')
    #hst_3d does not have 110 photometry
    hst3d_phot['F110_mag']=np.nan
    hst3d_phot['F110_mag_er']=np.nan
    #combine flags into one flag
    flgs=hst3d_phot[['use_phot_x', 'f_cover', 'f_flagged', 'f_negative']].values
    hst3d_phot['flags']= pd.Series([i for i in flgs])

    
    hst3d_phot['survey']='HST3D'
    wisp_phot['survey']='WISP'
    wisp_phot=wisp_phot.rename(columns={'EXTRACTION_FLAG':'flags'})
    
    #read in the spectral indices
    aegis=pd.read_pickle(INDICES+'aegis.pkl')
    goods=pd.read_pickle(INDICES+'goods.pkl')
    uds=pd.read_pickle(INDICES+'uds.pkl')
    cosmos=pd.read_pickle(INDICES+'cosmos.pkl')
    wisps_ids=pd.read_pickle(INDICES+'wisps.pkl')
    
    #combine them into one
    combined_ids=pd.concat([aegis, goods, uds, wisps_ids])
    #add indices as columns not dictionaries
    ids=pd.DataFrame([x for x in combined_ids.Indices])
    for x in ids.columns: combined_ids[x]= ids[x]

    print (combined_ids)
    def drop_ascii_from_name(s): 
        if not s.startswith('Par'): return (s.split('.1D.ascii')[0])
        else: return s
        
    
    indices=combined_ids.drop(labels='Indices', axis=1)
    indices['Names']=indices['Names'].apply(drop_ascii_from_name)
    
    #rename some columns
    indices=indices.rename(columns={'Names':'grism_id', 'Snrs':'cdf_snr'})
    
    #combined_photometry (the order matters: HST3D+WISPP
    comb_phot=pd.DataFrame()
    grism_ids=hst3d_phot['grism_id'].append(wisp_phot['grism_id'])
    comb_phot['grism_id']=grism_ids
    
    for flt in ['110', '140', '160']:
        mag_tuple1=hst3d_phot[['F'+flt+'_mag', 'F'+flt+'_mag_er']].apply(tuple, axis=1)
        mag_tuple2=wisp_phot[['NIMCOS_F'+flt+'W', 'NIMCOS_F'+flt+'W_ER']].apply(tuple, axis=1)
        mags=mag_tuple1.append(mag_tuple2)
        comb_phot['F'+flt]=mags
        
    ras=hst3d_phot['ra_x'].append(wisp_phot['RA'])
    decs=hst3d_phot['dec_x'].append(wisp_phot['DEC'])
        
    comb_phot['RA']=ras
    comb_phot['DEC']=decs
    comb_phot['survey']=hst3d_phot['survey'].append(wisp_phot['survey'])
    comb_phot['flags']=hst3d_phot['flags'].append(wisp_phot['flags'])
    
    
    master_table=pd.merge(indices, comb_phot, on='grism_id')
    
    print (indices.grism_id)
    print (comb_phot.grism_id)
    
    # I probably lost tons of objects with grism id ='0000'
    print (master_table.shape, comb_phot.shape, indices.shape )
    #measure line and std chi-square
    #df=master_table.grism_id.apply(compare_to_both)
    #df['x']=df.spex_chi/df.line_chi
    #df['F_x']=df.x.apply(f_test)

    #save the result
    #master_table=master_table.join(df)

    #drop the spectrum column because it makes the file heavier
    #master_table=master_table.drop(columns='spectra')

    master_table.to_hdf(COMBINED_PHOTO_SPECTRO_FILE, key='all_phot_spec_data')

    #make the cut 

    return 
	
if __name__=="__main__":
	combined_wisp_hst_catalogs()
	


