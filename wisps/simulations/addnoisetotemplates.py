
from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial
import splat
import wisps
from wisps import make_spt_number
import pandas as pd
import numpy as np

def add_noise_to_spectrum(sp, snr):
    #if I propose a larger SNR don't do anything to save time
    sp.reset()
    sp_old=sp.spectral_type
    sp.add_noise(snr, nsample=1, recompute_indices= True)
    f_test={"f_test": sp.f_test, 'line_chi': sp.line_chi, 'name': sp.name, 'spex_chi': sp.spex_chi, \
    'spt_new': sp.spectral_type, 'sp_old': sp_old, 'dof': sp.dof}
    res_dict= {**sp.snr, **sp.indices, **f_test}
    sp.reset()
    return res_dict
    
def add_multiple_noises(sp, noises):
    res=list(map(lambda x: add_noise_to_spectrum(sp, x), noises))
    return res

def add_noise_to_spectra(nsample=10, filein=wisps.LIBRARIES+'/ydwarfs.pkl', fileout=wisps.LIBRARIES+'/ydwarfs_plus_noise.pkl') :
    spectra=pd.read_pickle(filein)
    snrs=10**np.random.uniform(-1,3,(len(spectra), nsample))
    iterables=([spectra, snrs])

    method=partial(add_multiple_noises)
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]

    df=pd.DataFrame.from_records(results)
    df.to_pickle(fileout, protocol=2)

if __name__ =='__main__':
    #add_noise_to_spectra(nsample=500)
    add_noise_to_spectra(nsample=500, filein=wisps.LIBRARIES+'/subdwarfs.pkl', fileout=wisps.LIBRARIES+'/subdwarfs_plus_noise.pkl')