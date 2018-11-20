# -*- coding: utf-8 -*-

"""
goal: measure all indices by saving all spectra as objects then extracting a table of parameters in a different file
(might take forever but once it's done, it's done

"""

from .initialize import *
import glob
from .spectrum_tools import Spectrum
import glob
import os
from scipy import stats
import pandas as pd
import splat 
from tqdm import tqdm

#initialize standards 
splat.initializeStandards()
from os.path import expanduser
homedir = expanduser("~")

#get all paths, this should change on the remote computer 
path_to_wisps=REMOTE_FOLDER+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/*/1dspectra/*a_g141_*'
path_to_3d=REMOTE_FOLDER+'*/*/1D/*/*'
import pandas as pd

def get_all_paths():
    if not os.path.isfile('~/file_paths.txt'):
        paths=pd.DataFrame(np.append(glob.glob(path_to_wisps), glob.glob(path_to_3d)))[0]
        paths.to_csv('~/file_paths.txt')
    else:
        paths=pd.read_csv('~/file_paths.txt')[0]
    print (paths)
    return paths

def fit_a_line(wave, flux, noise):
    """
    Fit a line, returns a chi-square
    """
    m, b, r_value, p_value, std_err = stats.linregress(wave,flux)
    line=m*wave+b
    chisqr=np.nansum((flux-line)**2/noise**2)
    #print (chisqr)
    return line, chisqr

def fit_both(sp):
    s=sp.splat_spectrum
    s.trim([1.15, 1.65])
    line, chi=fit_a_line(s.wave.value, s.flux.value, s.noise.value)
    spt, spexchi=splat.classifyByStandard(s, return_statistic=True, fit_ranges=[[1.15, 1.65]], plot=False)
    result=pd.Series({'spex_chi':spexchi, 'line_chi':chi, 'spt': spt})
    return result


def test():
    #test if everything is working perfectly 
    p=get_all_paths()[0]
    sp=Spectrum(filepath=p)
    res=fit_both(sp)
    ##

    f = open(homedir+"/testfile.txt","a+") 
    f.write('{} \t {} \t {} \t{} \t{} \n'.format(sp.name, sp.snr, res.spex_chi, res.line_chi, res.spt))
    f.close()
    return 

def make_measurments(nbr=10):
    ps=get_all_paths().sample(n=nbr)
    print (ps)
    for p in tqdm(ps):
        try:
            sp=Spectrum(filepath=p)
            res=fit_both(sp)
            f = open(homedir+"/testfile.txt","a+")
            f.write('{} \t {} \t {} \t{} \t{} \t{} \n'.format(sp.name, sp.snr, res.spex_chi, res.line_chi, res.spt, sp.indices))
            f.close()
        except:
            pass
    return 

