# -*- coding: utf-8 -*-

"""
goal: measure all indices by saving all spectra as objects then extracting a table of parameters in a different file
(might take forever but once it's done, it's done

"""

from .initialize import *
import glob
import pickle
from .spectrum_tools import Spectrum
from tqdm import tqdm


def get_all_paths(**kwargs):
    survey=kwargs.get('survey', 'wisps')
    paths=[]
    if survey=='wisps':
        paths.append(glob.glob(REMOTE_FOLDER+'wisps/*/Spectra/*.dat'))

    if survey in ['aegis', 'cosmos', 'uds', 'goods']:
        paths.append(glob.glob(REMOTE_FOLDER+'/'+survey+'/*/1D/ASCII/*.ascii'))
        
    #print (paths)
<<<<<<< HEAD
    return np.array(paths[0])[:200]
=======
    return np.array(paths[0])
>>>>>>> smaller
    
def get_spectra(survey, **kwargs):
    spectra=[]
    paths=get_all_paths(survey=survey)
<<<<<<< HEAD
    print (paths)
    for p in tqdm((paths)):
        try:spectra.append(Spectrum(filepath=p))
        except: continue
=======
    print (len(paths))
    #split into chunks of 20,000
    chunks=np.array_split(paths, int(len(paths)/30000))
    for ck in chunks:
        for p in tqdm((paths)):
            try:
                print (p)
                spectra.append(Spectrum(filepath=p))
            except: 
                continue
>>>>>>> smaller
        
    #create a doictionary with spectra in them, will be used to retrieve spectra   	
    return spectra

if __name__=="__main__":
	#get_all_paths(survey='goods')
	get_spectra('goods')