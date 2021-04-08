
##adds survey parameters such as magnitude etc. to things

from .initialize import *
from tqdm import tqdm
from scipy.interpolate import interp1d

import wisps
from .initialize import SELECTION_FUNCTION, SPGRID
from wisps import drop_nan
from astropy.coordinates import SkyCoord
#import pymc3 as pm

from .core import  HS, MAG_LIMITS, Rsun, Zsun, custom_volume, SPGRID
import wisps.simulations as wispsim
from .binaries import make_systems
import numba
import dask
from scipy.interpolate import griddata
import wisps.simulations as wispsim
import pickle
from multiprocessing import Pool

POINTINGS=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')


#some re-arragments because the limiting distance depends on the pointing
dist_arrays=pd.DataFrame.from_records([x.dist_limits for x in POINTINGS]).applymap(lambda x:np.vstack(x).astype(float))

#ignore pymc, ignore pre-computed distances
 
POINTING_POINTING_NAMES= dict(zip([x.name for x in POINTINGS], POINTINGS))
#BAYESIAN_DISTANCES_VOLUMES=np.load(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', allow_pickle=True)
corr_pols=wisps.POLYNOMIAL_RELATIONS['mag_limit_corrections'] 

#imports
#----------------------

#constants
Rsun=wispsim.Rsun
Zsun=wispsim.Zsun

spgrid=SPGRID
#-----------------------


PNTS=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
pnt_names=[x.name for x in  PNTS]

#print (pnts[0].survey)
COORDS=SkyCoord([p.coord for p in PNTS ])
galc=COORDS.transform_to('galactic')

LBS=np.vstack([[x.coord.galactic.l.radian,x.coord.galactic.b.radian] for x in PNTS ])

LS=galc.l.radian
BS=galc.b.radian

#wispsim.make_pointings()
@numba.jit(nopython=True)
def fit_snr_exptime(ts, mag, d, e, f):
    return d*mag+e*np.log(ts/1000)+f

@numba.jit(nopython=True)
def mag_unc_exptime_relation( mag, t, m0, beta, a, b):
    sigma_min = 3.e-3
    tref = 1000.
    #m0, beta, a, b= params
    return ((t/tref)**-beta)*(10**(a*(mag-m0)+b))

def probability_of_selection(spt, snr):
    """
    probablity of selection for a given snr and spt
    """
    ref_df=SELECTION_FUNCTION.dropna()
    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
    interpoints=np.array([ref_df.spt.values, ref_df.logsnr.values]).T
    return griddata(interpoints, ref_df.tot_label.values , (spt, np.log10(snr)), method='linear')



def compute_effective_numbers(model, h):
    #DISTANCES=pd.DataFrame(pd.read_pickle(wisps.OUTPUT_FILES+'/cdf_distance_tables.pkl')[h])
    ##given a distribution of masses, ages, teffss
    ## based on my polynomial relations and my own selection function
    DISTANCE_SAMPLES=pd.read_pickle(wisps.OUTPUT_FILES+'/distance_samples{}.gz'.format(h))
    
    volumes=np.vstack([np.nansum(list(x.volumes[h].values())) for x in POINTINGS]).flatten()
    volumes_cdf=np.cumsum(volumes)/np.nansum(volumes)
    pntindex=np.arange(0, len(POINTINGS))
    names=np.array([x.name for x in POINTINGS])
    exptimes_mag=np.array([x.imag_exptime for x in POINTINGS ])
    exptime_spec= np.array([x.exposure_time for x in POINTINGS])
    

    syst=make_systems(model_name=model,  bfraction=0.2, nsample=5e4, recompute=True)

    
    #mask_array= np.logical_and(syst['system_spts']).flatten()
    spts=(syst['system_spts']).flatten()
    print ('----------------------------')
    print (model, h)
    print ('how many ......... {}'.format(len(spts)))
    mask= np.logical_and( spts>=17, spts<=41)
    spts=spts[mask]
    spt_r=np.round(spts)

    pntindex_to_use=wisps.random_draw(pntindex, volumes_cdf, nsample=len(spts)).astype(int)
    pnts=np.take(names, pntindex_to_use)
    exps= np.take(exptimes_mag, pntindex_to_use)
    exp_grism= np.take(exptime_spec, pntindex_to_use)

   
    #LONGS=(BAYESIAN_DISTANCES_VOLUMES['ls'][h]).flatten()
    #LATS=(BAYESIAN_DISTANCES_VOLUMES['bs'][h]).flatten()


    #retrieve key by key, let's see ho long it takes to run
    spt_r=np.floor(spts).astype(int)
    dists_for_spts= np.array([np.random.choice(DISTANCE_SAMPLES[k][idx]) for idx, k in zip(pntindex_to_use, spt_r)])
    #rs= pnt_distances[:,1][pntindex_to_use]
    #zs= pnt_distances[:,2][pntindex_to_use]



    #@np.vectorize
    #def match_dist_to_spt(spt, idxn):
    """
    one to one matching between distance and spt
    to avoid all sorts of interpolations or binning
    """
    #assign distance
    #spt_r=np.floor(spt)
    #d=np.nan
    #r=np.nan
    #z=np.nan
    #if (spt_r in DISTANCE_WITHIN_LIMITS_BOOLS.keys()):
    #bools=[(DISTANCE_WITHIN_LIMITS_BOOLS[k]) for x in spt_r][idxn]
    #dist_array=((BAYESIAN_DISTANCES_VOLUMES[h]['distances'])[idxn])#[bools]
    #rs=((BAYESIAN_DISTANCES_VOLUMES[h]['rs'])[idxn])#[bools]
    #zs=((BAYESIAN_DISTANCES_VOLUMES[h]['zs'])[idxn])#[bools]
    #draw a distance
    #if len(dist_array[bools]) <= 0 : 
    #    pass
    #else: 
    #   bidx=np.random.choice(len(dist_array[bools]))
    #    d= (dist_array[bools])[bidx]
    #    r=(rs[bools])[bidx]
    #    z=(zs[bools])[bidx]
    #return dist_array, rs, zs


    #polynomial relations
    relabsmags=wisps.POLYNOMIAL_RELATIONS['abs_mags']
    relsnrs=wisps.POLYNOMIAL_RELATIONS['snr']

    #print (relabsmags)
    #print (relsnrs)
   

    #add pointings
  

    #get distances withing magnitude limits

    #dbools=[DISTANCE_WITHIN_LIMITS_BOOLS[k] for k in spt_r]

    #assign distances using cdf-inversion
    #pnt_distances=  np.vstack([draw_distances(x, 1e5, h) for x in tqdm(POINTINGS)])
    #pnt_distances= (DISTANCES[names].values)#np.vstack([draw_distances(x, 1e5, h) for x in tqdm(POINTINGS)])
    #dists_for_spts=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['distances']).flatten()[pntindex_to_use]#[dbools]
    #rs=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['rs']).flatten()[pntindex_to_use]#[dbools]
    #zs=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['zs']).flatten()[pntindex_to_use]#[dbools]
    


    #distance_index= np.random.choice(np.arange(len(dist_array), len(spts)))

    #dists_for_spts= dist_array[distance_index]
    #rs=rs_array[distance_index]
    #zs=rs_array[distance_index]

    
    #compute magnitudes absolute mags
    f110s= np.random.normal((relabsmags['F110W'][0])(spts), relabsmags['F110W'][1])
    f140s= np.random.normal((relabsmags['F140W'][0])(spts), relabsmags['F140W'][1])
    f160s= np.random.normal((relabsmags['F160W'][0])(spts), relabsmags['F160W'][1])
    #compute apparent magnitudes
    appf140s0=f140s+5*np.log10(dists_for_spts/10.0)
    appf110s0=f110s+5*np.log10(dists_for_spts/10.0)
    appf160s0=f160s+5*np.log10(dists_for_spts/10.0)

    #print ('shape .....{}'.format(exps))
    
    #add magnitude uncertainities
    f110_ers=  mag_unc_exptime_relation(appf110s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F110']))
    f140_ers=  mag_unc_exptime_relation(appf140s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F140']))
    f160_ers=  mag_unc_exptime_relation(appf160s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F160']))

    appf110s= np.random.normal(appf110s0, f110_ers)
    appf140s= np.random.normal(appf140s0, f140_ers)
    appf160s= np.random.normal(appf160s0, f160_ers)

    #snrjs=10**np.random.normal( (relsnrs['snr_F140W'][0])(appf140s),relsnrs['snr_F140W'][1])
    #print (exp_grism)
    snrjs110= 10**(fit_snr_exptime(  exp_grism, appf110s, *list(MAG_LIMITS['snr_exp']['F110'])))
    snrjs140= 10**(fit_snr_exptime(  exp_grism, appf140s, *list(MAG_LIMITS['snr_exp']['F140'])))
    snrjs160= 10**(fit_snr_exptime(  exp_grism, appf160s, *list(MAG_LIMITS['snr_exp']['F160'])))

    snrjs= np.nanmin(np.vstack([snrjs110, snrjs140, snrjs160]), axis=0)

    sl= probability_of_selection(spts, snrjs)

    #comput the rest from the survey
    #dict_values={model: {h: {}, 'age': None, 'teff': None, 'spts': None}}
    #dict_values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
    #dict_values[model][h]={}
    #dict_values[model]={}
    #dict_values.update({model: {h:{}}})
    #print (model)
    #print (dict_values.keys())
    #print (np.nanmax(dict_values[model]['age']))
    #print (np.nanmax(syst['system_age'][~np.isnan((syst['system_spts']).flatten())]))
    #print (model)
    #print 
    #dict_values[model]['spts']=spts
    #dict_values[model]['teff']=syst['system_teff'][mask]
    #dict_values[model]['age']=

    morevals={'f110':f110s, 'f140':f140s, 'f160':f160s, 'd':dists_for_spts,  'appf140':appf140s,  
    'appf110':appf110s,  'appf160':appf160s, 'sl':sl, 'pnt':pnts, 'age':syst['system_age'][mask],
    'teff': syst['system_teff'][mask], 'spts': spts, 'f110_unc':  f110_ers, 'f140_unc':  f140_ers, 'f160_unc':  f160_ers,
    'snrj110':  snrjs110, 'snrj140':  snrjs140, 'snrj160':  snrjs160, 'snrj': snrjs} 



    #assert len(spts) == len(pnts)
    #assert len(f110s) == len(pnts)

    #dict_values[model][h].update(morevals)


    simdf=pd.DataFrame.from_records(morevals).rename(columns={'dist':'d', 
        'snrj': 'snr', 'slprob': 'sl', 'spts': 'spt', 'pnt': 'pntname'})
    
    
    
    simdf['pnt']=simdf.pntname.apply(lambda x: np.array(PNTS)[pnt_names.index(x)])
    
    
    #corrts0=

    mag_limits=pd.DataFrame.from_records(simdf.pnt.apply(lambda x: x.mag_limits).values)

    assert len(mag_limits)==len(appf140s0)


    flags0=simdf.appf110 >= mag_limits['F110']+(corr_pols['F110W'][0])(simdf.spt)
    flags1=simdf.appf140 >= mag_limits['F140']+(corr_pols['F140W'][0])(simdf.spt)
    flags2=simdf.appf160 >= mag_limits['F160']+(corr_pols['F160W'][0])(simdf.spt)
    flags3= simdf.snr <3.

    flags=np.logical_or.reduce([flags0,flags1, flags2, flags3])

    cutdf=(simdf[~flags]).reset_index(drop=True)
    #cutdf=simdf
    print ('Before cut {}'.format(len(simdf)))
    print ('After cut {}'.format(len(cutdf)))
    cutdf.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut.h5', key=str(model)+str('h')+str(h)+'F110_corrected')



    #import pickle

    #with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
    #        pickle.dump(dict_values,file)
    #return 
    
def get_all_values_from_model(model, hs):
    """
    For a given set of evolutionary models obtain survey values
    """
    #obtain spectral types from modelss
    for h in hs:
        compute_effective_numbers(model, h)

    #syst=make_systems(model_name=model, bfraction=0.2)
    #spts=(syst['system_spts']).flatten()
    #comput the rest from the survey
    #dict_values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
    #dict_values[model]['spts']=wisps.drop_nan(spts)
    #dict_values[model]['teff']=syst['system_teff'][~np.isnan(spts)]
    #dict_values[model]['age']=syst['system_age'][~np.isnan(spts)]
    #for h in tqdm(hs):
    #    dict_values[model][h]={}
    #    dict_values[model][h].update(compute_effective_numbers(wisps.drop_nan(spts),SPGRID, h))

    #import pickle
    #with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
    #        pickle.dump(dict_values,file)

    #del dict_values

def simulation_outputs(**kwargs):
    """
    Purpose:compute number densities
    """
    recompute=kwargs.get("recompute", False)
    hs=kwargs.get("hs", wispsim.HS)

    #recompute for different evolutionary models
    get_all_values_from_model('burrows2001', hs)
    get_all_values_from_model('baraffe2003', hs)
    get_all_values_from_model('saumon2008', hs)
    get_all_values_from_model('marley2019', hs)
    get_all_values_from_model('phillips2020', hs)
