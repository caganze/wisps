#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__= 'caganze'

#def meaure_indices()
from .initialize import *
from .spectrum_tools import *

import splat
import splat.plot as splt
from matplotlib.backends.backend_pdf import PdfPages

#short names for visually confirmed ugly spectra 
FORBIDDEN_LIST=['J0148+1202',  'J0331+4130', 'J0338-4409', 'J0343+3155',
' J0344+3204', '0344+3200', 'J0344+3156', 'J0344+3203',
'J0345+3205','J0419+2712', 'J0435+2254','J0438+2519',
'J0448-4432', 'J0448-4432', 'J0501-3341', 'J0512-2949',
'J0610-2151','J0621+6558','J0624-1045', 'J0628-0959',
'J0815+1041','J0822+1700', 'J0935+0019','J0950+0540',
'J1004+5023','J1050-2940','J1132-3018','J1132-3018',
'J1132-3018','J1132-3018','J1132-3018','J1132-3018',
'J1132-3018','J1132-3018', 'J1132-3018','J1132-3018',
'J1138-1314','J1209-3041','J1211-2821','J1224-2744',
'J1257-0204','J1303+2351','J1312+0051','J1317-1427',
'J1325-2128', 'J1420-1752', 'J1423+0116','J1629+1415',
'J1642-2355','J1642-2355','J1659+3515','J1726-1158',
'J1729+4352','J1829+5032','J1839-3744','J1924+5506',
'J1932+0133', 'J1932-3921', 'J1945-4149', 'J2001-3805',
'J2024-3422', 'J2028+6925', 'J2034+6727','J2151-3349']

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
        t=_load_and_save_spex_indices(OUTPUT_FILES+'/'+str(stype)+'spex_sample.pkl' , reload=True)
    return t 

def create_wisp_spectrum(filename):
    try:
        splat_spectrum=splat.getSpectrum(filename=filename)[0]
        #put it on the wisp resolution
        splat_spectrum.toInstrument('WFC3-G141')
        s=Spectrum(wave=splat_spectrum.wave.value,
                         flux=splat_spectrum.flux.value,
                         noise=splat_spectrum.noise.value)
        s.classify_by_standard
        return [s, splat_spectrum]
    except:
        return [None, None]

def plot_sp_sp(s, a, shortname):

    std=splat.STDS_DWARF_SPEX[splat.typeToNum(s.spectral_type[0])]
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
