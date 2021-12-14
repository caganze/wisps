import popsims
import numpy as np
import matplotlib.pyplot as plt
import wisps
import pandas as pd
import wisps.simulations as wispsim
from tqdm import tqdm 
import astropy.units as u
import numba
from scipy.interpolate import griddata


def probability_of_selection(spt, snr):
    """
    probablity of selection for a given snr and spt
    """
    ref_df=wispsim.SELECTION_FUNCTION.dropna()
    #self.data['spt']=self.data.spt.apply(splat.typeToNum)
    interpoints=np.array([ref_df.spt.values, ref_df.logsnr.values]).T
    return griddata(interpoints, ref_df.tot_label.values , (spt, np.log10(snr)), method='linear')

def get_snr(exp_grism, appf110s, appf140s, appf160s):
    #print (exp_grism)
    snrjs110= 10**(fit_snr_exptime(  exp_grism, appf110s, *list(wispsim.MAG_LIMITS['snr_exp']['F110'])))
    snrjs140= 10**(fit_snr_exptime(  exp_grism, appf140s, *list(wispsim.MAG_LIMITS['snr_exp']['F140'])))
    snrjs160= 10**(fit_snr_exptime(  exp_grism, appf160s, *list(wispsim.MAG_LIMITS['snr_exp']['F160'])))
    #assign upper and lo limits 
    snr_bool_up= np.logical_or.reduce([ appf110s >25,  appf140s >25,  appf160s>24])
    snr_bool_do= np.logical_or.reduce([ appf110s <15,  appf140s <15,  appf160s>15])

    snrjs= np.nanmin(np.vstack([snrjs110, snrjs140, snrjs160]), axis=0)
    
    return snrjs
    
def format_maglimits(wisp_limits):

    return {'WFC3_F110W':[16, wisp_limits['F110']],\
           'WFC3_F140W':[16, wisp_limits['F140']],\
           'WFC3_F160W':[16,wisp_limits['F160']]}


def make_cuts(df, dcts, expt):
    snr=get_snr(expt, df.WFC3_F110W.values, df.WFC3_F140W.values, df.WFC3_F160W.values)
    bools0=np.logical_or.reduce([df[k]< dcts[k][1] for k in dcts.keys()])
    return df[np.logical_and(bools0, snr>=3)]

def get_average_distance_limits(p, cuts):
    p.mag_limits=cuts
    return dict(pd.DataFrame(p.distance_limits).applymap(lambda x: \
                    x[1]).apply(lambda x: np.nanmedian(x), axis=1))


@numba.jit(nopython=True)
def fit_snr_exptime(ts, mag, d, e, f, m0):
    return d*(mag-m0)+e*np.log10(ts/1000)+f

@numba.jit(nopython=True)
def mag_unc_exptime_relation( mag, t, m0, beta, a, b):
    tref = 1000.
    #m0, beta, a, b= params
    return ((t/tref)**-beta)*(10**(a*(mag-m0)+b))

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
    app_er=  mag_unc_exptime_relation(app.values, df['exp_image'].values, *list(wispsim.MAG_LIMITS['mag_unc_exp'][mag_key]))

    df['app{}'.format(mag_key)]= np.random.normal(app, app_er)
    df['app{}'.format(mag_key)+'er']=app_er
    
    return df

def add_abs_mags(df):
    get_absmags_hst_filters(df, 'F110')
    get_absmags_hst_filters(df, 'F140')
    get_absmags_hst_filters(df, 'F160')
    #add magnitude uncertainities

    return df


def get_galactic_quantities():

    scaleH=900.
    scaleL=3600.

    thin_points=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
    distance_limits= thin_points[0].dist_limits


    coords=[x.coord for x in thin_points]
    pnt_names=[x.name for x in  thin_points]


    points=[popsims.Pointing(coord=p.coord, name=p.name) for p in tqdm(thin_points)]

 
    volumes={}

    distances={}

    for s in wispsim.SPGRID:
        volumes[s]={}
        distances[s]={}
        for p in tqdm(points):
            volumes[s][p.name] = popsims.volume_calc(p.coord.galactic.l.radian,\
                                   p.coord.galactic.b.radian,
                                    distance_limits[s][-1], distance_limits[s][0],scaleH, scaleL, \
                                   kind='exp')
            distances[s][p.name]= p.draw_distances(distance_limits[s][1]*0.5, 2*distance_limits[s][0], \
                scaleH, scaleL, nsample=1000)
            
    import pickle
    
    with open(wisps.OUTPUT_FILES+'/thick_disk_volumes.pkl', 'wb') as file:
        pickle.dump(volumes, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(wisps.OUTPUT_FILES+'/thick_disk_distances.pkl', 'wb') as file:
        pickle.dump(distances, file, protocol=pickle.HIGHEST_PROTOCOL)


def run_all():
    #load in some precomputed values
    corr_pols=wisps.POLYNOMIAL_RELATIONS['mag_limit_corrections'] 
    thin_points=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
    names=[x.name for x in thin_points]
    pntindex= np.arange(len(thin_points))
    volumes=pd.read_pickle(wisps.OUTPUT_FILES+'/thick_disk_volumes.pkl')
    DISTANCE_SAMPLES=pd.read_pickle(wisps.OUTPUT_FILES+'/thick_disk_distances.pkl')


    #compute total cumulative volumes
    tot_volumes_by_pointing=abs(np.nansum([[volumes[s][k] for k in  names] for s in wispsim.SPGRID], axis=0))
    tot_volumes_by_spt=abs(np.nansum([[volumes[s][k] for k in  names] for s in wispsim.SPGRID], axis=1))
    volumes_cdf= np.cumsum( tot_volumes_by_pointing)/np.nansum(   tot_volumes_by_pointing)

    #load in data from evolutionary models
    data=popsims.make_systems(model='baraffe2003', bfraction=0.2,\
                            mass_age_range= [0.01, 0.15, 8., 13.0],\
                                nsample=1e6,
                                recompute=True)

    #remove early types 
    spts= (data['spt'].values).flatten()
    mask= np.logical_and( spts>=17, spts<=41)
    df= data[mask].reset_index(drop=True)

    #assign distances
    spts=spts[mask]
    spt_r=np.round(spts)

    #assign pointings based on contributions to the tottal volumes
    pntindex_to_use=wisps.random_draw(pntindex, volumes_cdf, nsample=len(spts)).astype(int)
    pnts=np.take(thin_points, pntindex_to_use)
    pnts_names=np.take(names, pntindex_to_use)

    exptimes_mag=np.array([x.imag_exptime for x in  thin_points])
    exptime_spec= np.array([x.exposure_time for x in thin_points])
    exps= np.take(exptimes_mag, pntindex_to_use)
    exp_grism= np.take(exptime_spec, pntindex_to_use)

    #assign distance based on pointing and 
    #retrieve key by key, let's see ho long it takes to run
    spt_r=np.floor(spts).astype(int)
    dists_for_spts= np.array([np.random.choice(DISTANCE_SAMPLES[k][idx]) for idx, k in tqdm(zip(pnts_names, spt_r))])
    
    df['spt_r']=spt_r
    df['dist']=dists_for_spts
    df['thin_pointing']= pnts
    df['exp_image']=exps
    df['exp_grism']=exp_grism

    #compute magnitudes and snr
    df= add_abs_mags(df)
    snrjs110= 10**(fit_snr_exptime(  df.exp_grism.values, df.appF110.values, *list(wispsim.MAG_LIMITS['snr_exp']['F110'])))
    snrjs140= 10**(fit_snr_exptime(  df.exp_grism.values, df.appF140.values, *list(wispsim.MAG_LIMITS['snr_exp']['F140'])))
    snrjs160= 10**(fit_snr_exptime(  df.exp_grism.values, df.appF160.values, *list(wispsim.MAG_LIMITS['snr_exp']['F160'])))

    #assign upper and lo limits 
    snr_bool_up= np.logical_or.reduce([ df.appF110.values >25,  df.appF140.values >25,  df.appF160.values>24])
    snr_bool_do= np.logical_or.reduce([ df.appF110.values <15,  df.appF140.values <15,  df.appF160.values<15])


    snrjs= np.nanmin(np.vstack([snrjs110, snrjs140, snrjs160]), axis=0)

    #replace by 1000 or 1
    snrjs[snr_bool_up]=10**2.7
    snrjs[snr_bool_do]=1.

    sl= probability_of_selection(spts, snrjs)

    df['snrj']= snrjs
    df['sl']= sl
    print (df.columns)

    #make cuts based on magnitude limits
    mag_limits=pd.DataFrame.from_records(df.thin_pointing.apply(lambda x: x.mag_limits).values)
    flags0=df.appF110 >= mag_limits['F110']+(corr_pols['F110W'][0])(df.spt)
    flags1=df.appF140 >= mag_limits['F140']+(corr_pols['F140W'][0])(df.spt)
    flags2=df.appF160 >= mag_limits['F160']+(corr_pols['F160W'][0])(df.spt)
    flags3= df.snrj <3.

    flags=np.logical_or.reduce([flags0,flags1, flags2, flags3])

    cutdf=(df[~flags]).reset_index(drop=True)
    #save
    cutdf.to_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut_thick_disk.h5',\
     key=str('baraffe2003')+'F110_corrected')
    del cutdf

    
    return 

if __name__=='__main__':
    get_galactic_quantities()
    run_all()


