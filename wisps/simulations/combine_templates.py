##purpose: combines object spectral type and then reclassify them
##scaled to their absolute magnitudes

import splat
import wisps
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import splat.empirical as spe
import popsims

from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial

def proper_classification(sp):
    """
    Uses splat.classifyByStandard to classify spectra using spex standards
    """ 
    #sp.slitpixelwidth=1
    #sp.slitwidth=1
    #sp.toInstrument('WFC3-G141')

    wsp= wisps.Spectrum(wave=sp.wave.value, 
                           flux=sp.flux.value,
                          noise=sp.noise.value,
                        contam= np.ones_like(sp.noise.value))

    val=wisps.classify(wsp, stripunits=True)
    return val

def get_absolute_mag_j2mass(sptype):
    #spt=wisps.make_spt_number(sptype)
    spt=wisps.make_spt_number(sptype[0])
    mag=None #forget about kirkpatrick 
    #mag=spe.typeToMag(spt, '2MASS J', ref='dupuy2012')[0]
    if spt >20:
        mag=popsims.relations.absolute_mag_j(spt, ref='kirkpatrick2021', syst='mko')[0]
    else:
        mag=spe.typeToMag(spt, 'MKO J', ref='dupuy2012')[0]
    return mag

def combine_two_spectra(sp10, sp20):
    """
    sp1 and sp2 are splat objects
    """	

    
    sp1=sp10.splat_spectrum
    sp2=sp20.splat_spectrum
    
    #absj0=(wisps.absolute_magnitude_jh(wisps.make_spt_number(sp10.spectral_type[0]))[1]).flatten()[0]
    #absj1=(wisps.absolute_magnitude_jh(wisps.make_spt_number(sp20.spectral_type[0]))[1]).flatten()[0]

    #using kirkpatrick relations
    absj0=get_absolute_mag_j2mass(sp10.spectral_type)
    absj1=get_absolute_mag_j2mass(sp20.spectral_type)

    try:
        #luxCalibrate(self,filt,mag
        sp1.fluxCalibrate('2MASS J', absj0)
        sp2.fluxCalibrate('2MASS J', absj1)

        #print (sp1.wave, sp1.flux)    
        sp3=sp1+sp2

        print ("mags{}{} spts{}{} ".format(absj0, absj1,sp10.spectral_type[0], sp20.spectral_type[0] ))
        return {'primary': [sp10.spectral_type, sp20.spectral_type], 
        'system': proper_classification(sp3),
        'spectrum': sp3}
    except:
        return {}

def combine_two_spex_spectra(sp1, sp2):
    try:
        #first of all classifyByStandard
        spt1= splat.typeToNum(splat.classifyByStandard(sp1))
        spt2=splat.typeToNum(splat.classifyByStandard(sp2))
        
        #using kirkpatrick relations
        absj0=get_absolute_mag_j2mass(spt1)
        absj1=get_absolute_mag_j2mass(spt2)

        #luxCalibrate(self,filt,mag
        sp1.fluxCalibrate('2MASS J', absj0)
        sp2.fluxCalibrate('2MASS J', absj1)

        sp3= sp1+sp2
        spt3= splat.typeToNum(splat.classifyByStandard(sp3))
        print ("mags{}{} spts{}{}{} ".format(absj0, absj1,spt1, spt2, spt3 ))
        return {'primary_type': spt1,
                'secondary_type': spt2,
                'system_type': spt3,
                'system_wave_flux_noise': [sp3.wave.value, sp3.flux.value, sp3.noise.value],
                'primary_wave_flux_noise': [sp1.wave.value, sp1.flux.value, sp1.noise.value],
                'secondary_wave_flux_noise':[sp2.wave.value, sp2.flux.value, sp2.noise.value],
                }
    except:
        return {}


def make_binaries_spex():
    fl= '/volumes/LaCie/popsimsdata/spectral_templates_data.h5'
    tpls= pd.read_hdf(fl, key='singles')
    templates=list(tpls.spectra)
    iterables=list(np.array([(x, y) for x, y in tqdm(combinations(templates, 2))]).T)

    print (templates)
    #hjk

    method=partial(combine_two_spex_spectra)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    
    df=pd.DataFrame.from_records(results)

    return df.to_hdf(fl, key='binaries')

def make_binaries_wisp():
    ##
    tpls=pd.read_pickle(wisps.OUTPUT_FILES+'/l_t_dwarfs_spex.pkl')
    templates=list(tpls.spectra)
    iterables=list(np.array([(x, y) for x, y in tqdm(combinations(templates, 2))]).T)

    print (templates)
    #hjk


    method=partial(combine_two_spectra)
    with ThreadPoolExecutor(max_workers=10) as executor:
    	futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    
    df=pd.DataFrame.from_records(results)
    df.to_pickle(wisps.OUTPUT_FILES+'/binary_templates.pkl')


if __name__=='__main__':
    #make_binaries_wisp()
    make_binaries_spex()
