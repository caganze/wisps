#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import wisps
import wisps.simulations as wispsim
import pandas as pd
import splat
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import numba
from tqdm import tqdm
import splat.empirical as spem
import wisps.simulations.effective_numbers as ef 
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
import matplotlib as mpl
import splat.empirical as spe
import dask
#from dask.distributed import Client, progress
#dask.config.set({'scheduler.work-stealing': True, 'allowed-failures': 999})
#from dask import dataframe as dd 
import itertools

pnts=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
corr_pols=wisps.POLYNOMIAL_RELATIONS['mag_limit_corrections'] 
#cands=pd.read_pickle(wisps.LIBRARIES+'/candidates.pkl')
#tab=wisps.Annotator.reformat_table(cands)
pnt_names=[x.name for x in pnts]
sgrid=wispsim.SPGRID
import multiprocessing 



def bin_by_spt_bin(sp_types, number):
    ranges=[[17, 20], [20, 25], [25, 30], [30, 35], [35, 40]]
    numbers=[]
    for r in ranges:
        idx= np.logical_and((r[0]<=sp_types), (r[1]>sp_types))
        numbers.append(np.nansum(number[idx]))
    return numbers

def iswithin_mag_limits(mags, pnt, s):
    #mgs is a dictionary
    flags=[]
    for k in pnt.mag_limits.keys():
        if k =='F110' and pnt.survey =='hst3d':
            flags.append(False)
        else:
            corrt=(corr_pols['F110W'][0])(s)#-0.25
            #corrt=np.nanmedian([ (corr_pols['F160W'][0])(s),  (corr_pols['F110W'][0])(s),  (corr_pols['F140W'][0])(s)])
            flags.append(mags[k] < pnt.mag_limits[k]+corrt)
    return np.logical_or.reduce(flags)



def get_pointing(grism_id):
    if grism_id.startswith('par'):
        pntname=grism_id.lower().split('-')[0]
    else:
        pntname=grism_id.lower().split('-g141')[0]
    loc=pnt_names.index(pntname)
    return np.array(pnts)[loc]



def compute_simulated_numbers(hidx, model='saumon2008', selection='prob'):
    #an index in accordance with the scale height
    
    data=(ef.simulation_outputs()[model])[hidx]
    #print (data)

    #df=dd.from_pandas(data, npartitions=3)

    simdf=pd.DataFrame()
    simdf['spt']=(ef.simulation_outputs()[model])['spts'].flatten()
    simdf['teff']=(ef.simulation_outputs()[model])['teff'].flatten()
    simdf['age']=(ef.simulation_outputs()[model])['age'].flatten()
    simdf['slprob']=data['sl']
    simdf['snr']=data['snrj']
    simdf['appF140']=data['appf140']
    simdf['appF110']=data['appf110']
    simdf['appF160']=data['appf160']
    simdf['pntname']=data['pnt']
    simdf['dist']=data['d']

    
    simdf['pnt']=simdf.pntname.apply(lambda x: np.array(pnts)[pnt_names.index(x)])
    
    simmgs=simdf[['appF140', 'appF110', 'appF160']].rename(columns={"appF110": "F110", 
                                                                    "appF140": "F140",
                                                                    "appF160": "F160"}).to_dict('records')
    flags=[iswithin_mag_limits(x, y, z) for x, y,z in zip(simmgs,  simdf.pnt.values, simdf['spt'].values )]
    
    
    cutdf=(simdf[flags]).reset_index(drop=True)

    cutdf.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut.h5', key=str(model)+str('h')+str(hidx)+'F110_corrected')
    #save this cutdf dataframe
    #NORM = 0.63*(10**-3)/ len(cutdf.teff[np.logical_and(cutdf.teff>=1650, cutdf.teff <=1800)])
    
    #NSIM=dict(zip(wispsim.SPGRID,np.zeros(len(wispsim.SPGRID))))
    #rounded spectral type
    #cutdf['spt_r']=cutdf.spt.apply(np.round)
    #for g in cutdf.groupby('spt_r'):
    #   NSIM[g[0]]=np.nansum((g[1]).slprob*NORM)


        
    return {}

def compute_with_dask():

    #client = Client(processes=True, threads_per_worker=2,
    #            n_workers=100, memory_limit='2GB',  silence_logs='error')

    #lazy_results = []
    #pool = multiprocessing.Pool(processes=2)

    #Distribute the parameter sets evenly across the cores
    func=lambda x, y: compute_simulated_numbers(y, model=x)

    paramlist=[(i, j)  for i, j in itertools.product(['saumon2008', 'baraffe2003', 'marley2019', 'phillips2020'], wispsim.HS)]
    res  = [func(x, y) for x,y in tqdm(paramlist)]
    
    #for model in ['saumon2008', 'baraffe2003', 'marley2019', 'phillips2020']:
    #    for idx, h in enumerate(wispsim.HS):
    #        lazy_result= dask.delayed(compute_simulated_numbers)(h, model=model)
    #        lazy_results.append(lazy_result)

    #nexpct=dask.compute(*lazy_results)

    nexpct=np.array(res)

    return nexpct


if __name__ =='__main__':

    ds = compute_with_dask()


