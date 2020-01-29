
##adds survey parameters such as magnitude etc. to things

from .initialize import *
from tqdm import tqdm
from scipy.interpolate import interp1d

import wisps
from .initialize import SELECTION_FUNCTION, SPGRID
from wisps import drop_nan

from .core import  HS
from .binaries import make_systems
import numba



BAYESIAN_DISTANCES_VOLUMES=np.load(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', allow_pickle=True)

pnts=wisps.OBSERVED_POINTINGS
#some re-arragments because the limiting distance depends on the pointing
dist_arrays=pd.DataFrame.from_records([x.dist_limits for x in pnts]).applymap(lambda x:np.vstack(x).astype(float))
DISTANCE_LIMITS={}
for s in SPGRID:
    DISTANCE_LIMITS[s]=dist_arrays[s].mean(axis=0)

def probability_of_selection(vals, method='idx_ft_label'):
    """
    probablity of selection for a given snr and spt
    """
    ref_df=SELECTION_FUNCTION
    spt, snr=vals
    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
    floor=np.floor(spt)
    floor2=np.log10(np.floor(snr))
    return np.nanmean(ref_df[method][(ref_df.spt==floor) &(ref_df.snr.apply(np.log10).between(floor2, floor2+.3))])

@np.vectorize
def selection_function(spt, snr):
    return  probability_of_selection((spt, snr))

def compute_effective_numbers(spts,SPGRID, h):
    ##given a distribution of masses, ages, teffss
    ## based on my polynomial relations and my own selection function
    spts=wisps.make_spt_number(spts)

    DISTANCE_WITHIN_LIMITS_BOOLS={}
    #LONGS=(BAYESIAN_DISTANCES_VOLUMES['ls'][h]).flatten()
    #LATS=(BAYESIAN_DISTANCES_VOLUMES['bs'][h]).flatten()
    POINTINGS=wisps.OBSERVED_POINTINGS

    for k in DISTANCE_LIMITS.keys():
        dx=BAYESIAN_DISTANCES_VOLUMES['distances'][h]
        DISTANCE_WITHIN_LIMITS_BOOLS[k]= dx < 5* np.max(DISTANCE_LIMITS[k][0])

    @np.vectorize
    def match_dist_to_spt(spt, idxn):
        """
        one to one matching between distance and spt
        to avoid all sorts of interpolations or binning
        """
        #round spt to nearest spectral ty


        "-------------"
        #new idea, distance is a not a function of spt
        #scracth that
        "-------------"
        #assign distance
        spt_r=np.floor(spt)
        d=np.nan
        if (spt_r in DISTANCE_WITHIN_LIMITS_BOOLS.keys()):
            bools=(DISTANCE_WITHIN_LIMITS_BOOLS[spt_r])[idxn]
            dist_array=((BAYESIAN_DISTANCES_VOLUMES['distances'][h])[idxn])
            if len(dist_array[bools]) <= 0 : pass
            else: d= np.random.choice(dist_array[bools])
        return d


    #polynomial relations
    rels=wisps.POLYNOMIAL_RELATIONS

   

    #add pointings
    volumes=np.vstack([np.nansum(list(x.volumes[h].values())) for x in POINTINGS]).flatten()
    volumes_cdf=np.cumsum(volumes)/np.nansum(volumes)
    pntindex=np.arange(0, len(POINTINGS))
    names=np.array([x.name for x in POINTINGS])
    pntindex_to_use=wisps.random_draw(pntindex, volumes_cdf, nsample=len(spts))
    pnts=names[pntindex_to_use]


    #dists_for_spts=np.random.choice(dists_to_use, len(spts))
    dists_for_spts= match_dist_to_spt(spts,  pntindex_to_use)

    
    #compute magnitudes absolute mags
    f110s= np.random.normal(rels['sp_F110W'](spts), rels['sigma_sp_F110W'])
    f140s= np.random.normal(rels['sp_F140W'](spts), rels['sigma_sp_F140W'])
    f160s= np.random.normal(rels['sp_F160W'](spts), rels['sigma_sp_F160W'])
    #compute apparent magnitudes
    appf140s=f140s+5*np.log10(dists_for_spts/10.0)
    appf110s=f110s+5*np.log10(dists_for_spts/10.0)
    appf160s=f160s+5*np.log10(dists_for_spts/10.0)
    
    snrjs=10**np.random.normal(np.array(rels['snr_F140W'](appf140s)),rels['sigma_log_f140'])

    sl= selection_function(spts, snrjs)


    return f110s, f140s, f160s, dists_for_spts, appf140s,  appf110s,  appf160s, snrjs, sl, pnts


def get_all_values_from_model(model, **kwargs):
    """
    For a given set of evolutionary models obtain survey values
    """
    #obtain spectral types from modelss
    syst=make_systems(model_name=model, bfraction=0.1)
    spts=(syst['system_spts']).flatten()
    hs=kwargs.get("hs", HS)
    #comput the rest from the survey
    f110s=[]
    f140s=[]
    f160s=[]
    dists=[]
    snrjs=[]
    phis=[]
    appf140s=[]
    appf110s=[]
    appf160s=[]
    sl_probs=[]
    pntings=[]
    for h in tqdm(hs):
         f110, f140, f160, dists_for_spts, appf140, appf110, appf160, snrj, sl_prob, p=compute_effective_numbers(spts,SPGRID, h)
         f110s.append(f110)
         f140s.append(f140)
         f160s.append(f160)
         dists.append(dists_for_spts)
         snrjs.append(snrj)
         appf140s.append(appf140)
         appf110s.append(appf110)
         appf160s.append(appf160)
         sl_probs.append(sl_prob)
         pntings.append(p)
         
    values={"f110": f110s, "f140": f140s, "hs": hs, "f160": f160s, "appf140s": appf140s,"appf110s": appf110s,
     "appf160s": appf160s, "dists":dists, "snrjs": snrjs, "spgrid": SPGRID, 'spts': spts,
     'sl_prob': np.array(sl_probs), 'pointing': pntings, 'mass': (syst['system_mass']).flatten(), 'age': (syst['system_age']).flatten()}

    return values

def simulation_outputs(**kwargs):
    """
    Purpose:compute number densities
    """
    recompute=kwargs.get("recompute", False)

    #recompute for different evolutionary models
    models=kwargs.get('models', ['saumon2008', 'baraffe2003', 'marley2019'])

    if recompute:
        dict_values={}
        for model in models: dict_values[model]= get_all_values_from_model(model)
        import pickle
        with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
            pickle.dump(dict_values,file)
    else:
        dict_values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
    return dict_values


