##purpose: combines object spectral type and then reclassify them
##scaled to their absolute magnitudes

import splat
import wisps
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm



from concurrent.futures import ThreadPoolExecutor, wait , ALL_COMPLETED
from  functools import partial

def combine_two_spectra(sp10, sp20):
	"""
	sp1 and sp2 are splat objects
	"""
	sp1=sp10.splat_spectrum
	sp2=sp20.splat_spectrum
	sp3=sp1+sp2
	types={}
	types={'primary': [sp10.spectral_type, sp20.spectral_type], 
	'system': splat.classifyByStandard(sp3,  comprange=[[1.2, 1.6]], dwarf=True,subdwarf=False,  statistic='chisqr')[0],
	'spectrum': sp3}
	return types

def make_binaries():
    ##
    tpls=pd.read_pickle(wisps.OUTPUT_FILES+'/l_t_dwarfs_spex.pkl')
    templates=np.random.choice(tpls,300)
    iterables=list(np.array([(x, y) for x, y in combinations(templates, 2)]).T)

    method=partial(combine_two_spectra)
    with ThreadPoolExecutor(max_workers=100) as executor:
    	futures=list(executor.map( method, *iterables, timeout=None, chunksize=10))

    results=[x for x in futures]
    
    df=pd.DataFrame.from_records(results)
    df.to_pickle(wisps.OUTPUT_FILES+'/binary_templates.pkl')


if __name__=='__main__':
	make_binaries()