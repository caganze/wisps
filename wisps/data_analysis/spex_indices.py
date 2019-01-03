#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__= 'caganze'

#def meaure_indices()
from .initialize import *
from .spectrum_tools import *

import splat
    
def spex_sample_ids(**kwargs):
    """
    This function loads indices measured for spectral standardards, templates, subdwarfs and esds either by reading the saved file or re-computing them
    input: flag specifying database, and whether to load from saved file
    outut: table of source name, spectral type and an array of 10 indices
    
    """
    stype= kwargs.get('stype', 'std')
    from_file=kwargs.get('from_file', True)
    #if prompted to read from file, return the indices (faster)
    if from_file:
        print ("reading from file")
        t=pd.read_pickle(OUTPUT_FILES+'/'+str(stype)+'spex_sample.pkl' )
    #sometimes files are empty, re-calculate the indices, save them 
    if not from_file:
        print ("calculating spectral indices")
        t=_load_and_save_spex_indices(stype,OUTPUT_FILES+'/'+str(stype)+'spex_sample.pkl' )
    return t 

def _load_and_save_spex_indices(stype, filename):
    
    """
    Avoiding to repeat some code
    
    """
    #load spectra
    
    dict_keys=['M5', 'M6', 'M7', 'M8', 'M9',
               'L0', 'L1', 'L2', 'L3', 'L4', 
               'L5','L6', 'L7', 'L8', 'L9', 
               'T0', 'T1', 'T2', 'T3', 'T4',
               'T5', 'T6', 'T7', 'T8', 'T9']
    if stype=='spex_sample':
        db= splat.searchLibrary(spt=[17, 39], vlm=True, giant=False )
        spectra= [splat.Spectrum(x) for x in db['DATA_FILE']]
        spts=np.array(db['SPTN'])
    if stype=='std':
        spectra=[splat.getStandard(x) for x in dict_keys]
        spts=[splat.typeToNum(x) for x in  dict_keys]
    if stype=='sd':
        db=splat.searchLibrary(spt=[17, 39], subdwarf=True)
        spectra= [splat.Spectrum(x) for x in db['DATA_FILE']]
        spts=np.array(db['SPTN'])
    if stype =='esd':
        spectra=[splat.core.getStandard(x, esd=True) for x in dict_keys]
    #print ( splat_spectra)
    names=[s.name for s in spectra]
    #print (names)
    filenames=[s.filename for s in spectra]
    
    #print (names)
    # for x in splat_spectra:
    #     try:
    #      x.plot()
    #     except:
    #         continue
    
    nempty=[i for i, x in enumerate(names) if x]
    wisp_spectra= [Spectrum(wave=s.wave.value, flux=s.flux.value, noise=s.noise.value ) for s in np.array(spectra)[nempty]]
    #classifyByStandard
    chis=[ splat.classifyByStandard(s.splat_spectrum, return_statistic=True, plot=False)[1] for s in wisp_spectra]
    #measure indices
    ids= pd.DataFrame([s.indices for s in wisp_spectra ])
    #get different measures of snrs
    snrs=[s.snr for s in wisp_spectra ]
    #save the file
    final_dict={'Names':names, 'Spts': np.array(spts)[nempty],  'Snr':snrs, 'Chis':chis, 'spectra': wisp_spectra}
    t= pd.DataFrame(final_dict)
    for k in ids.columns: t[k]=ids[k]
    t.to_pickle(filename)
    
    return t
