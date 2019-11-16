

from .initialize import *
from tqdm import tqdm
from scipy.interpolate import interp1d

import wisps
from .initialize import SELECTION_FUNCTION, SPGRID

from .core import  simulate_spts
import numba

SIMULATED_DIST=simulate_spts()


BAYESIAN_DISTANCES_VOLUMES=np.load(wisps.OUTPUT_FILES+'/bayesian_pointings.pkl', allow_pickle=True)
DISTANCE_LIMITS=np.load(wisps.OUTPUT_FILES+'/distance_limits.pkl',allow_pickle=True)
#bayesian distances

def drop_nan(x):
    x=np.array(x)
    return x[(~np.isnan(x)) & (~np.isinf(x)) ]


def interpolated_lf(spts, lumin):
    f = interp1d(spts, lumin)
    return f(SPGRID)

@numba.jit
def probability_of_selection(vals, method='f_test_label'):
    """
    probablity of selection for a given snr and spt
    """
    ref_df=SELECTION_FUNCTION
    spt, snr=vals
    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
    floor=np.floor(spt)
    floor2=np.log10(np.floor(snr))
    if floor2 < np.log10(3.):
        return 0.
    if floor2 > np.log10(2.0):
        return 1.

    if floor <17 or  floor >42:
        return 0.0

    else:
        return np.nanmean(ref_df[method][(ref_df.spt==floor) &(ref_df.snr.apply(np.log10).between(floor2, floor2+.3))])

@numba.vectorize(['float32(float32, float32)','float64(float64, float64)'])
def selection_function(spt, snr):
    return  probability_of_selection((spt, snr))

def compute_effective_numbers(spts,SPGRID, h):
    ##given a distribution of masses, ages, teffss
    ## based on my polynomial relations and my own selection function
    DISTANCE_WITHIN_LIMITS={}
    for k in DISTANCE_LIMITS.keys():
        dx=BAYESIAN_DISTANCES_VOLUMES['distances'][h]
        DISTANCE_WITHIN_LIMITS[k]= dx[np.logical_and(dx>DISTANCE_LIMITS[k][1]/2., dx< 10* DISTANCE_LIMITS[k][1])]
    
    @numba.vectorize("float64(float64)")
    def match_dist_to_spt(spt):
        """
        one to one matching between distance and spt
        to avoid all sorts of interpolations or binning
        """
        #round spt to nearest spectral ty


        "-------------"
        #new idea, distance is a not a function of spt
        #scracth that
        "-------------"
        spt_r=np.floor(spt)
        if spt_r in SPGRID:
            d= np.random.choice(DISTANCE_WITHIN_LIMITS[spt_r])
        else:
            d=np.nan
        return d

    ds=(BAYESIAN_DISTANCES_VOLUMES['distances'])[h]

    #group similar spts then draw distances
    #dists_for_spts=[]
    #for k in tqdm(DISTANCE_LIMITS.keys()):
    #    spts_to_use=np.logical_and(spts <=k, spts<k+1)
    #    upper, lower=DISTANCE_LIMITS[k]
    #    d_choose= ds[np.logical_and(ds>lower, ds<upper*2)] #only choose between dmin and twice dmax
    #    dists_for_spts.append(np.random.choice(d_choose, len(spts_to_use)))
        
    #polynomial relations
    rels=wisps.POLYNOMIAL_RELATIONS
    #effective volumes
    #assign distances
    #dis
    
    #dists_for_spts=np.random.choice(dists_to_use, len(spts))
    dists_for_spts= match_dist_to_spt(spts)
   
    
    #compute magnitudes absolute mags
    f110s= rels['sp_F110W'](spts)
    f140s= rels['sp_F140W'](spts)
    f160s= rels['sp_F160W'](spts)
    #compute apparent magnitudes
    appf140s=f140s+5*np.log10(dists_for_spts/10.0)
    appf110s=f110s+5*np.log10(dists_for_spts/10.0)
    appf160s=f160s+5*np.log10(dists_for_spts/10.0)
    #compute snr based on my relations
    #only use F140W for SNRJS
    #offset them by the scatter in the relation
    f140_snrj_scatter=rels['sigma_log_f140']
    snrjs=10**np.random.normal(np.array(rels['snr_F140W'](appf140s)), f140_snrj_scatter)
    #apply the selection function (this is the slow part)
    sl= selection_function(spts, snrjs)
    #the probabliy of smearing
    #vts=np.random.uniform(0, 100, len(dists_for_spts))
    #mus=proper_motion(vts, dists_for_spts)
    #smearing_p=probability_of_detection_smearing(mus)

    #group these by spt
    df=pd.DataFrame()
    df['spt']=spts
    #df['ps_sme']=smearing_p*sl #selection probability including proper motion
    df['appF140']=appf140s
    df['appF110']=appf110s
    df['appF160']=appf160s
    df['snr']=snrjs
    #round the spt for groupings
    df.spt=df.spt.apply(round)
    
    #make selection cuts 
    flag_snr=(df.snr > 3.0).values
    flag0=  (df.appF140.between( wisps.MAG_LIMITS['hst3d']['F140W'][1], wisps.MAG_LIMITS['hst3d']['F140W'][0])).values 
    flag1=  (df.appF110.between( wisps.MAG_LIMITS['wisps']['F110W'][1], wisps.MAG_LIMITS['wisps']['F110W'][0])).values
    flag2=  (df.appF110.between( wisps.MAG_LIMITS['hst3d']['F160W'][1], wisps.MAG_LIMITS['hst3d']['F160W'][0])).values

    flag= ~ np.logical_and.reduce((flag0, flag1, flag2, flag_snr))

    #cut the dataframe, these are the things you should select
    #the selection function is the number of things you select /number of things you should select
    #not the number of things you simulate

    

    #probablity of selection > 1.
    df['ps']=sl

    (df['ps'])[flag]=0.0
    
    #group by spt and sum
    phi0=[]
    phi0_spts=[]

    df_cut=df[flag]

    fractions=float(len(df)/len(df_cut))

    for g in df.groupby('spt'):
        phi0_spts.append(g[0])
        phi0.append(float(np.nansum(g[1].ps))*fractions)

    idx=[i for i, x in enumerate(phi0_spts) if x in SPGRID]

    #finally some luminosity function!
    phi=np.array(phi0)[idx]
    #return all the data

    return f110s, f140s, f160s, dists_for_spts, appf140s,  appf110s,  appf160s, snrjs, phi, df.ps.values


def simulation_outputs(**kwargs):
    """
    Purpose:compute number densities
    """
    hs=kwargs.get("hs", [100, 250, 275, 300, 325, 350, 1000])
    recompute=kwargs.get("recompute", False)

    if recompute:
         #BAYESIAN_DISTANCES= BAYESIAN_DISTANCES_VOLUMES['distances']
         BAYESIAN_VOLUMES= BAYESIAN_DISTANCES_VOLUMES['volumes']
         vols=[[vols[s] for s in SPGRID] for vols in BAYESIAN_VOLUMES]
         VOLUMES=np.nansum(np.array(vols), axis=0)*(4.1*(u.arcmin**2).to(u.radian**2))
         #for g, gr in enumerate(SPGRID):
         #    BAYESIAN_DICT[gr]=dict(zip( hs, BAYESIAN_DISTANCES[:,g,: ]))
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
         for h in tqdm(hs):
             spts=drop_nan(SIMULATED_DIST['spts'][0][:,0])
             f110, f140, f160, dists_for_spts, appf140, appf110, appf160, snrj, phi, sl_prob=compute_effective_numbers(spts,SPGRID, h)
             f110s.append(f110)
             f140s.append(f140)
             f160s.append(f160)
             dists.append(dists_for_spts)
             snrjs.append(snrj)
             phis.append(phi)
             appf140s.append(appf140)
             appf110s.append(appf110)
             appf160s.append(appf160)
             sl_probs.append(sl_prob)

             
         values={"f110": f110s, "f140": f140s, "hs": hs, "f160": f160s, "appf140s": appf140s,"appf110s": appf110s,
         "appf160s": appf160s, "dists":dists, "snrjs": snrjs, "n": phi, "spgrid": SPGRID, "vol": VOLUMES, 
         'sl_prob': np.array(sl_probs)}
         
         import pickle
         with open(wisps.OUTPUT_FILES+'/effective_numbers_from_sims', 'wb') as file:
             pickle.dump(values,file)
    else:
        values=pd.read_pickle(wisps.OUTPUT_FILES+'/effective_numbers_from_sims')
    return values


