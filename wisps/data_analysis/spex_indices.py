#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__= 'caganze'

#def meaure_indices()
from .initialize import *
from .spectrum_tools import *

import splat
import splat.plot as splt
from matplotlib.backends.backend_pdf import PdfPages

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
        t=_load_and_save_spex_indices(OUTPUT_FILES+'/'+str(stype)+'spex_sample.pkl' )
    return t 

def create_wisp_spectrum(filename):
    try:
        splat_spectrum=splat.getSpectrum(filename=filename)[0]
        s=Spectrum(wave=splat_spectrum.wave.value,
                         flux=splat_spectrum.flux.value,
                         noise=splat_spectrum.noise.value)
        s.classify_by_standard
        return [s, splat_spectrum]
    except:
        return [None, None]

def plot_sp_sp(s, a, shortname):

    std=splat.STDS_DWARF_SPEX[splat.typeToNum(s.spectral_type)]
    std.normalize(waverange=[1.1, 1.7])

    mask2=np.logical_and(std.wave.value>0.8, std.wave.value<2.5)
    mask=np.logical_and(s.wave>0.8, s.wave<2.5)

    a.plot(s.wave[mask], s.flux[mask], label=shortname)
    a.plot(std.wave.value[mask2], std.flux.value[mask2], linestyle='--', label='std')
    
    a.set_title("{} ".format(s.spectral_type ))
    a.legend()

def _load_and_save_spex_indices(fname, **kwargs):
    """
    save splat stuff
    """
    if kwargs.get('reload', False):
        splat_db=splat.searchLibrary(vlm=True, giant=False)
        ss=splat_db.DATA_FILE.apply(create_wisp_spectrum)
        spectra=np.vstack(ss.dropna().values)[:, 0]
        splat_spectra=np.vstack(ss.dropna().values)[:, 1]
        spectra=spectra[spectra != np.array(None)]

        splat_spectra=splat_spectra[splat_spectra != np.array(None)]
        legends=[x.shortname for x in splat_spectra]

        df=pd.DataFrame()
        df['splat']= splat_spectra
        df['wisps']=spectra
        df['shortname']=legends

        df.to_pickle(fname)

    else:
        df=pd.read_pickle(fname)

    #splt.plotBatch(list(splat_spectra), classify=True, legend=list(legends), output= OUTPUT_FIGURES+'/multipage_splat_stuff_2nd.pdf')
    with PdfPages(OUTPUT_FIGURES+'/multipage_splat_stuff_2nd.pdf') as pdf:

        for g in np.array_split(df, int(len(df)/4)):

            fig, ax=plt.subplots(ncols=2, nrows=2)
            plot_sp_sp(g.wisps.iloc[0], ax[0][0], g.shortname.iloc[0])
            plot_sp_sp(g.wisps.iloc[1], ax[0][1],  g.shortname.iloc[1])
            plot_sp_sp(g.wisps.iloc[2], ax[1][0],  g.shortname.iloc[2])
            plot_sp_sp(g.wisps.iloc[3], ax[1][1],  g.shortname.iloc[3])

            pdf.savefig() 
            plt.close()

    
    return df
