
##adds survey parameters such as magnitude etc. to things

#from .initialize import *
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import wisps
from wisps import drop_nan
from astropy.coordinates import SkyCoord
#import pymc3 as pm

from wisps.simulations import  HS, MAG_LIMITS, Rsun, Zsun, custom_volume, SELECTION_FUNCTION, SPGRID
import wisps.simulations as wispsim
#from .binaries import make_systems
import numba
import dask
from scipy.interpolate import griddata
import wisps.simulations as wispsim
import pickle
import popsims
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
def fit_snr_exptime(ts, mag, d, e, f, m0):
    return d*(mag-m0)+e*np.log10(ts/1000)+f

@numba.jit(nopython=True)
def mag_unc_exptime_relation( mag, t, m0, beta, a, b):
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


def get_distances_and_pointings(df, h):

    DISTANCE_SAMPLES=pd.read_pickle(wisps.OUTPUT_FILES+'/distance_samples{}'.format(h))
    volumes=np.vstack([np.nansum(list(x.volumes[h].values())) for x in POINTINGS]).flatten()
    volumes_cdf=np.cumsum(volumes)/np.nansum(volumes)
    pntindex=np.arange(0, len(POINTINGS))
    names=np.array([x.name for x in POINTINGS])
    exptimes_mag=np.array([x.imag_exptime for x in POINTINGS ])
    exptime_spec= np.array([x.exposure_time for x in POINTINGS])


    spt_r=np.round(df.spt.values)
    pntindex_to_use=wisps.random_draw(pntindex, volumes_cdf, nsample=len(spt_r)).astype(int)

    
    pnts=np.take(names, pntindex_to_use)
    pntings=np.take( np.array(POINTINGS),   pntindex_to_use)
    #print (  pntings)
    exps= np.take(exptimes_mag, pntindex_to_use)
    exp_grism= np.take(exptime_spec, pntindex_to_use)

    spt_r=np.floor(df.spt.values).astype(int)
    dists_for_spts= np.array([np.random.choice(DISTANCE_SAMPLES[k][idx]) for idx, k in zip(pntindex_to_use, spt_r)])

    df['dist']=  dists_for_spts
    df['pntname']= pnts
    df['pnt']=pntings#df.pntname.apply(lambda x: np.array(PNTS)[pnt_names.index(x)])
    df['exp_image']= exps
    df['exp_grism']=exp_grism

    return df


def get_snr_and_selection_prob(df):

    snrjs110= 10**(fit_snr_exptime(df['exp_grism'].values, df['appF110'].values, *list(MAG_LIMITS['snr_exp']['F110'])))
    snrjs140= 10**(fit_snr_exptime(df['exp_grism'].values, df['appF140'].values, *list(MAG_LIMITS['snr_exp']['F140'])))
    snrjs160= 10**(fit_snr_exptime(df['exp_grism'].values, df['appF160'].values, *list(MAG_LIMITS['snr_exp']['F160'])))

    df['snrj110']=snrjs110
    df['snrj140']= snrjs140
    df['snrjs160']= snrjs160

    df['snrj']=np.nanmin(np.vstack([snrjs110, snrjs140, snrjs160]), axis=0)
    df['slprob']=probability_of_selection(df.spt.values,  df['snrj'].values)

    return df
    

def get_absmags_hst_filters(df, mag_key):
    """
    get abs_mag

    """
    #load relations 
    relabsmags=wisps.POLYNOMIAL_RELATIONS['abs_mags']
    relcolors=wisps.POLYNOMIAL_RELATIONS['colors']
    binary_flag=df.is_binary.values

    #compute absolue magnitudes for singles
    res=np.ones_like(df.spt.values)*np.nan
    abs_mags_singles=np.random.normal((relabsmags[mag_key+'W'][0])(df.spt.values), relabsmags[mag_key+'W'][1])


    #for binaries, base this on their absolute J and H mag
    color_key='j_'+mag_key.lower()
    #if mag_key=='F160':
    #    color_key='h_f160'

    #colors=np.random.normal((relcolors[color_key][0])(df.spt.values), relcolors[color_key][1])
    #abs_mags_binaries=df['abs_2MASS_J']-colors
    abs_mag_primaries=np.random.normal((relabsmags[mag_key+'W'][0])(df.prim_spt.values) , relabsmags[mag_key+'W'][1])
    abs_mag_secondaries=np.random.normal((relabsmags[mag_key+'W'][0])(df.sec_spt.values) , relabsmags[mag_key+'W'][1])

    abs_mags_binaries=-2.5*np.log10(10**(-0.4* abs_mag_primaries)+10**(-0.4* abs_mag_secondaries))

    np.place(res, ~binary_flag, abs_mags_singles[~binary_flag])
    np.place(res, binary_flag, abs_mags_binaries[binary_flag])


    #absolute mag
    df['abs{}'.format(mag_key)]=res
    df['prim_abs{}'.format(mag_key)]=abs_mag_primaries
    df['sec_abs{}'.format(mag_key)]= abs_mag_secondaries
    #df['abs{}'.format(mag_key)]=abs_mags_singles

    #apparent mag
    app=res+5*np.log10(df.dist/10.0)
    app_er=  mag_unc_exptime_relation(app.values, df['exp_image'].values, *list( MAG_LIMITS['mag_unc_exp'][mag_key]))

    df['app{}'.format(mag_key)]= np.random.normal(app, app_er)
    df['app{}'.format(mag_key)+'er']=app_er
    return df


def compute_effective_numbers(model, h):
    """
    model: evol model
    h : scaleheights
    """

    df0=popsims.make_systems(model_name=model, bfraction=0.2,\
                            mass_age_range= [0.01, 0.15, 0.1, 8.0],\
                                nsample=int(1e6),
                                save=True)

    #drop nans in spt
    df0=(df0[~df0.spt.isna()]).reset_index(drop=True)
    mask= np.logical_and(df0.spt>=17, df0.spt<=41)
    df0=(df0[mask]).reset_index(drop=True)

    #assign distances and poiunts
    df0=get_distances_and_pointings(df0, h)



    #assign absolute mags
    df0=get_absmags_hst_filters(df0, 'F110')
    df0=get_absmags_hst_filters(df0, 'F140')
    df0=get_absmags_hst_filters(df0, 'F160')
    
    print(df0.keys())
    #add snr and selection probability
    df0=get_snr_and_selection_prob(df0)
    mag_limits=pd.DataFrame.from_records(df0.pnt.apply(lambda x: x.mag_limits).values)


    #make cuts
    flags0=df0.appF110 >= mag_limits['F110']+(corr_pols['F110W'][0])(df0.spt)
    flags1=df0.appF140 >= mag_limits['F140']+(corr_pols['F140W'][0])(df0.spt)
    flags2=df0.appF160 >= mag_limits['F160']+(corr_pols['F160W'][0])(df0.spt)
    flags3= df0.snrj <3.

    flags=np.logical_or.reduce([flags0,flags1, flags2, flags3])

    df0['is_cut']=flags

    df0.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut_binaries_updatedrelations.h5', key=str(model)+str(h)+str('spt_abs_mag'))





    #cutdf.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut.h5', key=str(model)+str('h'))

def compute_effective_numbers_old(model, h):
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
    

    syst=make_systems_nocombined_light(model_name=model,  bfraction=0.2, nsample=5e3, recompute=True)
    print (len(syst))

    
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

    #assign upper and lo limits 
    snr_bool_up= np.logical_or.reduce([ appf110s >25,  appf140s >25,  appf160s>24])
    snr_bool_do= np.logical_or.reduce([ appf110s <15,  appf140s <15,  appf160s>15])


    snrjs= np.nanmin(np.vstack([snrjs110, snrjs140, snrjs160]), axis=0)

    #replace by 1000 or 1
    snrjs[snr_bool_up]=10**2.7
    snrjs[snr_bool_do]=1.

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
    #print ('Before cut {}'.format(len(simdf)))
    #print ('After cut {}'.format(len(cutdf)))


    cutdf.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut.h5', key=str(model)+str(h)+str(h)+'F110_corrected')
    del cutdf



    #import pickle

    #with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
    #        pickle.dump(dict_values,file)
    return 
    
def get_all_values_from_model(model, hs):
    """
    For a given set of evolutionary models obtain survey values
    """
    #obtain spectral types from modelss
    for h in tqdm(hs):
        _= compute_effective_numbers(model, h)

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

if __name__=='__main__':
    """c
    Purpose:compute number densities
    """
    #recompute=kwargs.get("recompute", True)
    #hs=kwargs.get("hs", )

    #recompute for different evolutionary models
    get_all_values_from_model('burrows1997', wispsim.HS)
    get_all_values_from_model('burrows2001',  wispsim.HS)
    get_all_values_from_model('baraffe2003',  wispsim.HS)
    get_all_values_from_model('saumon2008',  wispsim.HS)
    get_all_values_from_model('marley2019',  wispsim.HS)
    get_all_values_from_model('phillips2020',  wispsim.HS)
