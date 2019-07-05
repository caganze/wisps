#my colormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import splat
import pandas as pd
import splat.empirical as spem
import statsmodels.nonparametric.kernel_density as kde
import numba

#################
splat.initializeStandards()
###############
from wisps.utils import memoize_func

#customize the interpolation function
from scipy.interpolate import interp1d

class extrap1d(object):
    """
    custom extrapolator object
    """
    def __init__(self, **kwargs):
        self.interpolator=kwargs.get('interp', None) #must interp1d object
        if not (self.interpolator is None):
            xs=self.interpolator.x
            ys=self.interpolator.y
        else:
            xs=None
            ys=None
        self.xs=xs
        self.ys=ys
    
    def pointwise(self, x):
        if x < self.xs[0]:
            return 0.0
        elif x > self.xs[-1]:
            return float(self.ys[-1])
        else:
            return self.interpolator(x)

    def ufunclike(self, nx):
        nx=np.array(nx)
        nx=nx.reshape(len(nx), 1)
        return np.apply_along_axis(self.pointwise, 1, nx)

def my_color_map():
        colors1 = plt.cm.BuGn(np.linspace(0., 1, 256))
        colors2 = plt.cm.Purples(np.linspace(0., 1, 256))
        colors3 = plt.cm.cool(np.linspace(0., 1, 256))
        colors4 = plt.cm.Greens(np.linspace(0., 1, 256))
        colors = np.vstack((colors1+colors2)/2)
        colorsx = np.vstack((colors3+colors4)/2)
        return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors), mcolors.LinearSegmentedColormap.from_list('my_other_colormap', colorsx)

MYCOLORMAP, MYCOLORMAP2=my_color_map()

@memoize_func
def stats_kde(x, **kwargs):
    grid=np.arange(np.nanmin(x), np.nanmax(x))
    model=kde.KDEMultivariate(x, bw='normal_reference', var_type='c')
    return grid, model.cdf(grid), model.pdf(grid)

@numba.jit
def make_spt_number(spt):
    ##make a spt a number
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

def conv_to_galactic(ra, dec, d):
    '''
    Function to convert ra, dec, and distances into 
    Galactocentric coordinates R, theta, Z.
    From Loki
    '''

    const = pd.Series(CONSTANTS)
    r2d   = 180. / np.pi # radians to degrees

    # Check if (numpy) arrays
    if isinstance(ra, np.ndarray)  == False:
        ra  = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
    if isinstance(d, np.ndarray)   == False:
        d   = np.array(d).flatten()

    # Convert values to Galactic coordinates
    """
    # The SLOOOOOOOOOOW Astropy way
    c_icrs = SkyCoord(ra = ra*u.degree, dec = dec*u.degree, frame = 'icrs')  
    l, b = c_icrs.galactic.l.radian, c_icrs.galactic.b.radian
    """
    l, b = radec2lb(ra, dec)
    l, b = np.deg2rad(l), np.deg2rad(b)
    
    r    = np.sqrt( (d * np.cos( b ) )**2 + const.RSUN.value * (const.RSUN.value - 2 * d * np.cos( b ) * np.cos( l ) ) )
    t    = np.rad2deg( np.arcsin(d * np.sin( l ) * np.cos( b ) / r) )
    z    = const.ZSUN.value + d * np.sin( b - np.arctan( const.ZSUN.value / const.RSUN.value) )
    
    return np.array([r, t, z])

def radec2lb(ra, dec):

    '''
    Convert ra,dec values into Galactic l,b coordinates
    '''

    # Make numpy arrays
    if isinstance(ra, np.ndarray)  == False:
        ra = np.array(ra).flatten()
    if isinstance(dec, np.ndarray) == False:
        dec = np.array(dec).flatten()
        
    # Fix the dec values if needed, should be between (-90,90)
    dec[ np.where( dec > 90 )]  = dec[ np.where( dec > 90 )]  - 180
    dec[ np.where( dec < -90 )] = dec[ np.where( dec < -90 )] + 180
    
    psi    = 0.57477043300
    stheta = 0.88998808748
    ctheta = 0.45598377618
    phi    = 4.9368292465
    
    a    = np.deg2rad( ra ) - phi
    b    = np.deg2rad( dec )
    sb   = np.sin( b )
    cb   = np.cos( b )
    cbsa = cb * np.sin( a )
    b    = -1 * stheta * cbsa + ctheta * sb
    bo   = np.rad2deg( np.arcsin( np.minimum(b, 1.0) ) )
    del b
    a    = np.arctan2( ctheta * cbsa + stheta * sb, cb * np.cos( a ) )
    del cb, cbsa, sb
    ao   = np.rad2deg( ( (a + psi + 4. * np.pi ) % ( 2. * np.pi ) ) )

    return ao, bo

#@memoize_func
def distance(mags, spt):
    """
    mags is a dictionary of bright and faint mags

    set a bias 

    SET A RANK 110 FIRST'
    140 next, don't do a scatter
    """
    res=pd.Series()
    
    f110w=mags['F110W']
    f140w=mags['F140W']
    f160w=mags['F160W']

    res['absj'] = spem.typeToMag(splat.typeToNum(spt),'2MASS J',set='dupuy')[0]
    res['absh'] = spem.typeToMag(splat.typeToNum(spt),'2MASS H',set='dupuy')[0]

    for k in mags.keys():
        flt='NICMOS '+k
        #calculate the HST mag- Abs J offset of the standard
        sp = splat.STDS_DWARF_SPEX[spt]
        sp.fluxCalibrate('2MASS J',float(sp.j_2mass))
        mag, mag_unc = splat.filterMag(sp, flt)
        #calculate the mag of the standard in J and H
        magj, mag_unck = splat.filterMag(sp,'2MASS J', ab=True)
        magh, mag_unck = splat.filterMag(sp,'2MASS H', ab=True)
        #calculate the offset between HST filters and 2mass filters
        offsetj=magj-mag
        offset2=magh-mag
        #add that offset to the mag to find the j, h mag of the source
        source_j=mags[k]+offsetj
        source_h=mags[k]+offsetj
        #calculate the two distances
        distj=10.**((source_j-res['absj'])/5. + 1.)
        disth=10.**((source_h-res['absh'])/5. + 1.)
        #take the standard deviation
        res[str('dist')+k]=np.nanmean([distj, disth], axis=0)
        res[str('dist_er')+k]=np.nanstd([distj, disth], axis=0)

    return res
