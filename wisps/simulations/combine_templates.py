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
    val=wisps.classify(sp, stripunits=True)
    return val

def get_absolute_mag_h2mass(sptype):
    spt=wisps.make_spt_number(sptype)
    mag=None
    if spt >20:
        mag=absolute_mag_h(spt, ref='kirkpatrick2021', syst='2mass')[0]
    else:
        mag=absolute_mag_h(spt, ref='dupuy2012', syst='2mass')[0]
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
    absj0=get_absolute_mag_h2mass(sp10.spectral_type)
    absj1=get_absolute_mag_h2mass(sp20.spectral_type)



    try:
        sp1.fluxCalibrate('2MASS H',  absj0)
        sp2.fluxCalibrate('2MASS H',  absj1)
        
        sp3=sp1+sp2
        return {'primary': [sp10.spectral_type, sp20.spectral_type], 
        'system': proper_classification(sp3),
        'spectrum': sp3}
    except:
        return {}


def make_binaries():
    ##
    tpls=pd.read_pickle(wisps.OUTPUT_FILES+'/binary_spex.pkl')
    templates=tpls.spectra
    iterables=list(np.array([(x, y) for x, y in tqdm(combinations(templates, 2))]).T)


    method=partial(combine_two_spectra)
    with ThreadPoolExecutor(max_workers=100) as executor:
    	futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    
    df=pd.DataFrame.from_records(results)
    df.to_pickle(wisps.OUTPUT_FILES+'/binary_templates.pkl')


if __name__=='__main__':
	make_binaries()