
##adds survey parameters such as magnitude etc. to things

from .initialize import *
from tqdm import tqdm
from scipy.interpolate import interp1d

import wisps
from .initialize import SELECTION_FUNCTION, SPGRID
from wisps import drop_nan

from .core import  HS, MAG_LIMITS, Rsun, Zsun, custom_volume
from .binaries import make_systems
import numba
import dask
from scipy.interpolate import griddata


POINTINGS=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')



#some re-arragments because the limiting distance depends on the pointing
dist_arrays=pd.DataFrame.from_records([x.dist_limits for x in POINTINGS]).applymap(lambda x:np.vstack(x).astype(float))

#ignore pymc, ignore pre-computed distances
DISTANCE_LIMITS={}
#BAYESIAN_DISTANCES_VOLUMES=np.load(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', allow_pickle=True)

#---------------------------
def draw_distance_with_cdf(l, b, dmax, nsample, h):
    #draw distances using inversion of the cumulative distribution 
    #this is to avoid using pymc
    d=np.logspace(0, np.log10(dmax), int(nsample))
    cdfvals=np.array([custom_volume(l,b,0, dx, h) for dx in d])
    return wisps.random_draw(d, cdfvals/np.nanmax(cdfvals), int(nsample))


def draw_distances(pnt, nsample, h):
    #draw distances for each pointing separtely up to a 10000 pc
    l, b= pnt.coord.galactic.l.radian, pnt.coord.galactic.b.radian,
    dists=draw_distance_with_cdf(l, b, 4000, nsample, h)
    #get rs and zs
    xs=Rsun-dists*np.cos(b)*np.cos(l)
    ys=-dists*np.cos(b)*np.sin(l)
    rs=(xs**2+ys**2)**0.5 
    zs=Zsun+ dists * np.sin(b)
    return [dists, rs, zs]

for s in SPGRID:
    DISTANCE_LIMITS[s]=dist_arrays[s].mean(axis=0)

DISTANCE_LIMITS[42]=[np.nan, np.nan]
#def probability_of_selection(vals, method='tot_label'):
#    """
#    probablity of selection for a given snr and spt
#    """
#    ref_df=SELECTION_FUNCTION
#    spt, snr=vals
#    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
#    floor=np.floor(spt)
#    floor2=np.log10(np.floor(snr))
#    return np.nanmean(ref_df[method][(ref_df.spt==floor) &(ref_df.logsnr.between(floor2, floor2+.3))])


def probability_of_selection(spt, snr):
    """
    probablity of selection for a given snr and spt
    """
    ref_df=SELECTION_FUNCTION.dropna()
    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
    interpoints=np.array([ref_df.spt.values, ref_df.logsnr.values]).T
    return griddata(interpoints, ref_df.tot_label.values , (spt, np.log10(snr)), method='linear')

#@np.vectorize
#def selection_function(spt, snr):
#    return  probability_of_selection((spt, snr))

#@numba.jit(nopython=True)
def func_total(x, z, a, c, x0, b):
    #fit of magnitude and exposure time to get uncertainties
    return x0+ a*(x**c)+ b*(z)

def compute_effective_numbers(model, h):
    ##given a distribution of masses, ages, teffss
    ## based on my polynomial relations and my own selection function
    syst=make_systems(model_name=model, bfraction=0.2)
    spts=(syst['system_spts']).flatten()
    spts=wisps.drop_nan(spts)
    spts=wisps.make_spt_number(spts)

    DISTANCE_WITHIN_LIMITS_BOOLS={}
    #LONGS=(BAYESIAN_DISTANCES_VOLUMES['ls'][h]).flatten()
    #LATS=(BAYESIAN_DISTANCES_VOLUMES['bs'][h]).flatten()

    #for k in DISTANCE_LIMITS.keys():
    #    dx=(BAYESIAN_DISTANCES_VOLUMES[h])['distances']
    #    DISTANCE_WITHIN_LIMITS_BOOLS[k]= dx < 10* np.max(DISTANCE_LIMITS[k][0])


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
    volumes=np.vstack([np.nansum(list(x.volumes[h].values())) for x in POINTINGS]).flatten()
    volumes_cdf=np.cumsum(volumes)/np.nansum(volumes)
    pntindex=np.arange(0, len(POINTINGS))
    names=np.array([x.name for x in POINTINGS])
    exptimes=np.array([np.log10(x.exposure_time) for x in POINTINGS ])
    pntindex_to_use=wisps.random_draw(pntindex, volumes_cdf, nsample=len(spts))
    pnts=names[pntindex_to_use]
    exps= exptimes[pntindex_to_use]

    #get distances withing magnitude limits
    spt_r=np.floor(spts)

    #dbools=[DISTANCE_WITHIN_LIMITS_BOOLS[k] for k in spt_r]

    #assign distances using cdf-inversion
    pnt_distances= np.vstack([draw_distances(x, 1e4, h) for x in tqdm(POINTINGS)])
    #dists_for_spts=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['distances']).flatten()[pntindex_to_use]#[dbools]
    #rs=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['rs']).flatten()[pntindex_to_use]#[dbools]
    #zs=np.vstack(BAYESIAN_DISTANCES_VOLUMES[h]['zs']).flatten()[pntindex_to_use]#[dbools]
    dists_for_spts= pnt_distances[:,0][pntindex_to_use]
    rs= pnt_distances[:,1][pntindex_to_use]
    zs= pnt_distances[:,2][pntindex_to_use]



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
    
    #add magnitude uncertainities
    appf110s= np.random.normal(appf110s0, func_total(appf110s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F110'])))
    appf140s= np.random.normal(appf140s0, func_total(appf140s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F140'])))
    appf160s= np.random.normal(appf160s0, func_total(appf160s0, exps, *list( MAG_LIMITS['mag_unc_exp']['F160'])))

    snrjs=10**np.random.normal( (relsnrs['snr_F140W'][0])(appf140s),relsnrs['snr_F140W'][1])

    sl= probability_of_selection(spts, snrjs)

    #comput the rest from the survey
    dict_values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
    print (model)
    print (dict_values.keys())
    print (np.nanmax(dict_values[model]['age']))
    print (np.nanmax(syst['system_age'][~np.isnan((syst['system_spts']).flatten())]))
    #print (model)
    #print 
    dict_values[model]['spts']=wisps.drop_nan((syst['system_spts']).flatten())
    dict_values[model]['teff']=syst['system_teff'][~np.isnan((syst['system_spts']).flatten())]
    dict_values[model]['age']=syst['system_age'][~np.isnan((syst['system_spts']).flatten())]

    morevals={'f110':f110s, 'f140':f140s, 'f160':f160s, 'd':dists_for_spts, 'r':rs, 'z':zs, 'appf140':appf140s,  
    'appf110':appf110s,  'appf160':appf160s, 'snrj':snrjs, 'sl':sl, 'pnt':pnts}

    dict_values[model][h].update(morevals)

   
    import pickle

    with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
            pickle.dump(dict_values,file)
    
def get_all_values_from_model(model):
    """
    For a given set of evolutionary models obtain survey values
    """
    #obtain spectral types from modelss
    for h in HS:
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

    #recompute for different evolutionary models
    models=kwargs.get('models', ['saumon2008', 'baraffe2003' ,'marley2019', 'phillips2020'])

    if recompute:
        #dict_values={}
        for m in models: 
            get_all_values_from_model(m)
        #import pickle
        #with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
        #    pickle.dump(dict_values,file)
        return
    else:
        dict_values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
        return dict_values