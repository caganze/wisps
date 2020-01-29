

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

#proper pointing name
def get_proper_pointing(grism_id):
    grism_id=grism_id.lower()
    if grism_id.startswith('par'):
        return grism_id.split('-')[0]
    else:
        return grism_id.split('-g141')[0]


#constants
STARS=wisps.Annotator.reformat_table(datasets['stars'])
STARS['pointing']=STARS.grism_id.apply(get_proper_pointing)
STARS=STARS[STARS.snr1 >=3.]

Rsun=83000.
Zsun=27.
HS=[100,200, 250,300, 350,400, 600]

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
    """
    faint_dict={'F110W': (mag_limits['F110'], 0.0), 'F140W': (mag_limits['F140'], 0.0), 'F160W':(mag_limits['F160'], 0.0)}
    bright_dict={'F110W': (18., 0.0), 'F140W': (16., 0.0), 'F160W': (16., 0.0)}
    distances=[]
    if np.isnan([ x for x in mag_limits.values()]).all():
        return {}
    for s in SPGRID:
        dmaxs=wisps.distance(faint_dict, s)
        dmins=wisps.distance(bright_dict, s)
        dmx=np.nanmean([dmaxs['distF110W'],dmaxs['distF140W'], dmaxs['distF160W']])
        dmin=np.nanmean([dmins['distF110W'],dmins['distF140W'], dmins['distF160W']])
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
    values=wisps.drop_nan(values)
    if len(values)<1:
        return np.nan
    if  np.equal.reduce(values):
        return  values.mean()
    if (len(values)>1) and (~np.equal.reduce(values)):
        kernel = kernel_density(values)
        height = kernel.pdf(values)
        mode_value = values[np.argmax(height)]
        return float(mode_value)

def get_mag_limit(pnt, key, mags):
    #fit for less than 50
    survey='wisps'
    maglt=np.nan
    if not pnt.name.lower().startswith('par'): 
        survey='hst3d'
        if key=='F110':
            return maglt

    if len(mags) < 50:
        magpol=MAG_LIMITS[survey][key][0]
        magsctt=MAG_LIMITS[survey][key][1]
        maglt=np.nanmean(np.random.normal(magpol(np.log10(pnt.exposure_time)), magsctt, 100))

    #use KDEs for more than 50
    if len(mags) >= 50:
        maglt=get_max_value(mags)
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
        self.observation_date=None

        #compute volumes after initialization
        if self.name is not None:
            df=STARS[STARS.pointing==self.name.lower()]
            self.exposure_time=(df['exposure_time']).values.mean()
            self.observation_date=(df['observation_date']).values
            for k in ['F110', 'F140', 'F160']:
                self.mags[k]=df[k].values
                self.mag_limits[k]= get_mag_limit(self, k, self.mags[k])

    def compute_volume(self):
        self.dist_limits=compute_distance_limits(self.mag_limits)
        for h in HS:
            self.volumes[h]=computer_volume(self, h)


def make_pointings():
    
    obs=pd.read_csv(wisps.OUTPUT_FILES+'/observation_log.csv')
    obs=obs.drop(columns=['Unnamed: 0']).drop_duplicates(subset='POINTING').reindex()
    obs.columns=[x.lower() for x in obs.columns]

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

    pnts=[make_pointing(ra, dec, survey, name) for ra, dec, survey, name in tqdm(zip(ras, decs, surveys, obs.pointing.values))]
    pnts=[x for x in pnts if x.dist_limits]

    import pickle

    output_file=wisps.OUTPUT_FILES+'/pointings.pkl'
    with open(output_file, 'wb') as file:
        pickle.dump(pnts,file)
