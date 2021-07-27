

################################
# population simulations routines

##############################

#imports


from .initialize import *
from astropy.coordinates import SkyCoord
import scipy.integrate as integrate
from scipy import stats
from ..utils.tools import drop_nan, splat_teff_to_spt,kernel_density
from ..data_sets import datasets
from tqdm import tqdm
import splat.simulate as spsim
import splat.evolve as spev
import wisps
from scipy import stats

#proper pointing name
def get_proper_pointing(grism_id):
    grism_id=grism_id.lower()
    if grism_id.startswith('par'):
        return grism_id.split('-')[0]
    else:
        return grism_id.split('-g141')[0]

#print (MAG_LIMITS)
#constants
big_file=wisps.get_big_file()
#starswisp=stars[ big_file.class_starstars.grism_id.str.startswith('par')]
#starshst3d=stars[(~ stars.grism_id.str.startswith('par')) & (stars.star_flags !=2.) ]
#starshst3d=stars[(~ stars.grism_id.str.startswith('par'))  & ((stars.snr1 > 10 )& (stars.class_star > 0.6))]
#starshst3d=stars[(~ stars.grism_id.str.startswith('par'))]

STARS= (big_file[ big_file.mstar_flag !=0]).reset_index(drop=True)
#STARS['pointing']=STARS.grism_id.apply(get_proper_pointing)
STARS=wisps.Annotator.reformat_table(STARS[STARS.snr1>=3.]).reset_index(drop=True)


del big_file

Rsun=8300.
Zsun=27.
HS=[100, 150,200, 250, 300, 350, 400, 450, 500, 600,800, 1000]

#redefine magnitude limits by taking into account the scatter for each pointing 
#use these to compute volumes

#REDEFINED_MAG_LIMITS={'F110':    23.054573, 'F140':    23.822972, 'F160' :   23.367867}

#-------------------------------------------
def density_function(r, z, h=300.):
    
    """
    A custom juric density function that only uses numpy arrays for speed
    All units are in pc
    """
    l = 2600. # radial length scale of exponential thin disk 
    zpart=np.exp(-abs(z-Zsun)/h)
    rpart=np.exp(-(r-Rsun)/l)
    return zpart*rpart


def custom_volume(l,b,dmin, dmax, h):
    nsamp=1000
    ds = np.linspace(dmin,dmax,nsamp)
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    rh0=density_function(rd, zd,h=h )
    val=integrate.trapz(rh0*(ds**2), x=ds)
    return val

def get_accurate_relations(x, rel, rel_unc):
    #use monte-carlo error propgation
    vals=np.random.normal(rel(x), rel_unc, 100)
    return np.nanmean(vals)


def compute_distance_limits(mag_limits):
    """
    computes distance limits based on limiting mags
    take the mininum distance of the the three because that incorporates every simulated
    """
    bright_dict={'F110W': [16., 0.0], 'F140W': [16., 0.0], 'F160W': [16., 0.0]}
    distances=[]
    if np.isnan([ x for x in mag_limits.values()]).all():
        return {}
    #add new correction term for each subtype
    corr_pols=wisps.POLYNOMIAL_RELATIONS['mag_limit_corrections'] 
    for s in SPGRID:
        #add corrections to key but only use F110W corrections
        #corrt=np.nanmedian([ (corr_pols['F160W'][0])(s),  (corr_pols['F110W'][0])(s),  (corr_pols['F140W'][0])(s)])
        corrt0=(corr_pols['F110W'][0])(s)#+0.5
        corrt1=(corr_pols['F140W'][0])(s)#+0.5
        corrt2=(corr_pols['F160W'][0])(s)#+0.5
        #corrt=0.0
        faint_dict={'F110W': [mag_limits['F110']+corrt0, 0.0], 
        'F140W': [mag_limits['F140']+corrt1, 0.0],
        'F160W':[mag_limits['F160']+corrt2, 0.0]}

        dmaxs=wisps.distance(faint_dict, s, 0.0)
        dmins=wisps.distance(bright_dict, s, 0.0)
        #just use minimum
        dmx=np.nanmedian([dmaxs['distF110W'],dmaxs['distF140W'], dmaxs['distF160W']])
        dmin=np.nanmedian([dmins['distF110W'],dmins['distF140W'], dmins['distF160W']])
        distances.append([dmx, dmin])
    return  dict(zip(SPGRID, distances))
    
def computer_volume(pnt, h):
        """
        given area calculate the volume
        """
        volumes={}
        if pnt.dist_limits:
            lb=[pnt.coord.galactic.l.radian,pnt.coord.galactic.b.radian]
            for k in SPGRID:
                volumes[k]= np.array(custom_volume(lb[0],lb[1],  pnt.dist_limits[k][1], pnt.dist_limits[k][0], h))
        return volumes

def get_max_value(values):
    #make a mask
    #mask=np.logical_or.reduce([np.isnan(values), values <15, values >25])
    values=wisps.drop_nan(values)
    #values=values[~mask]
    if len(values)<1:
        return np.nan
    if np.equal.reduce(values):
        return np.nanmean(values)
    if len(values)>=1:
        #kernel = wisps.kernel_density(values)
        #kernel= stats.gaussian_kde(distr, bw_method=0.35)
        #height = kernel.pdf(values)
        #kernel = wisps.kernel_density(values)
        kernel= stats.gaussian_kde(distr, bw_method=0.1)
        height = kernel.pdf(np.linspace(10, 30, 1000))
        mode_value = values[np.argmax(height)]
        print (mode_value)
        return float(mode_value)



def get_mag_limit(pnt, key, mags):
    #fit for less than 50
    maglt=np.nan
    survey= 'wisps'
  
    #leave 3d hst alone
    if (not pnt.name.lower().startswith('par')): 
        #survey='hst3d'
        if (key=='F110'): 
            return maglt

        else:
            if pnt.imag_exptime<800:
                maglt= 22.5
            else:
                maglt= 23.0
    #things above 50 objects

    if (len(mags) >= MAG_LIMITS['ncutoff']): 
        maglt=get_max_value(mags)

    #if (pnt.name.lower().startswith('par')): 
    #also fits things brighter than 12
    if ((len(mags) < MAG_LIMITS['ncutoff']) or (maglt <12)) & (pnt.name.lower().startswith('par')):
        #print (maglt)
        magpol=MAG_LIMITS['mag_limits'][survey][key][0]
        magsctt=MAG_LIMITS['mag_limits'][survey][key][1]
        maglt=np.random.normal(magpol(np.log10(pnt.imag_exptime)), magsctt)

    return maglt

   


class Pointing(object):
    ## a pointing object making it easier to draw samples
    def __init__(self, **kwargs):
        #only input is the direction
        self.coord=kwargs.get('coord', None)
        self.survey=kwargs.get('survey', None)
        self.name=kwargs.get('name', None)
        self.mags={}
        self.mag_limits={}
        self.dist_limits={}
        self.volumes={}
        self.exposure_time=None
        self.exposure_times=None
        self.observation_date=None
        self.snr1=None
        self.number_of_sources={}
        self.mags_unc={}
        self.imag_exptime=None

        #compute volumes after initialization
        if self.name is not None:
            df=STARS[STARS.pointing.str.lower()==self.name.lower()]
            self.exposure_time=(df['exposure_time']).values.mean()
            self.exposure_times=(df['exposure_time']).values
            self.observation_date=(df['observation_date']).values
            self.snr1=df.snr1.values
            self.imag_exptime=np.nanmean(df.expt_f140w.values)
            for k in ['F110', 'F140', 'F160']:
                self.mags[k]=df[k].values
                self.mags_unc
                self.mag_limits[k]= get_mag_limit(self, k, self.mags[k])
                self.number_of_sources[k]= len(self.mags[k])

            del df

    def compute_volume(self):
        self.dist_limits=compute_distance_limits(self.mag_limits)
        for h in HS:
            self.volumes[h]=computer_volume(self, h)


def make_pointings():
    
    obs=pd.read_csv(wisps.OUTPUT_FILES+'/observation_log.csv')
    obs=obs.drop(columns=['Unnamed: 0']).drop_duplicates(subset='POINTING').reindex()
    obs.columns=[x.lower() for x in obs.columns]
    print (len(obs))

    def make_pointing(ra, dec, survey, name):
        coord=SkyCoord(ra=ra*u.deg,dec=dec*u.deg )
        p=Pointing(coord=coord, survey=survey, name=name)
        p.compute_volume()
        return p

    def get_survey(pointing):
        if pointing.startswith('par'):
            return 'wisps'
        else:
            return 'hst3d'

    ras=obs['ra (deg)']
    decs=obs['dec(deg)']
    surveys=obs.pointing.apply(get_survey)

    pnts=[make_pointing(ra, dec, survey, name) for ra, dec, survey, name in zip(ras, decs, surveys, obs.pointing.values)]
    #if 
    print ('missing magnitude limits' , [x.name for x in pnts if not x.dist_limits])
    pnts=[x for x in pnts if x.dist_limits]

    #assert len(pnts)==533

    import pickle

    output_file=(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(pnts,file)
