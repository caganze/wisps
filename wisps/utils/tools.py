#my colormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import splat
import pandas as pd
import splat.empirical as spem
import statsmodels.nonparametric.kernel_density as kde

#################
splat.initializeStandards()
###############


def my_color_map():
        colors1 = plt.cm.BuGn(np.linspace(0., 1, 256))
        colors2 = plt.cm.Purples(np.linspace(0., 1, 256))
        colors = np.vstack((colors1+colors2)/2)
        return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

MYCOLORMAP=my_color_map()

def stats_kde(x, **kwargs):
    grid=np.arange(np.nanmin(x), np.nanmax(x))
    model=kde.KDEMultivariate(x, bw='normal_reference', var_type='c')
    return grid, model.cdf(grid), model.pdf(grid)

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

def distance(mags, spt):
    """
    mags is a dictionary of bright and faint mags
    """
    res=pd.Series()
    
    f110w=mags['F110W']
    f140w=mags['F140W']
    f160w=mags['F160W']
    #convert these fuckers to 2MASS J, H, K
    #get colo
    
    res['absj'] = spem.typeToMag(splat.typeToNum(spt),'2MASS J',set='dupuy')[0]
    res['absh'] = spem.typeToMag(splat.typeToNum(spt),'2MASS H',set='dupuy')[0]

    for k in mags.keys():
        sp = splat.STDS_DWARF_SPEX[spt]
        sp.fluxCalibrate('2MASS J',float(sp.j_2mass))
        #sp.filterMag('NICMOS F140W')
        c1=sp.filterMag('2MASS J')[0]-sp.filterMag(k)[0]
        c2=sp.filterMag('2MASS H')[0]-sp.filterMag(k)[0]
        
        res[str('rel')+k+str('j')] = mags[k]+c1
        res[str('rel')+k+str('h')] = mags[k]+c2

        dist1=10**((res[str('rel')+k+str('j')]-res['absj'])/5.+1.)#[0]
        dist2=10**((res[str('rel')+k+str('h')]-res['absh'])/5.+1.)#[0]
        res[str('dist')+k]=np.sqrt(dist1**2+dist2**2)
        res[str('dist_er')+k]=np.nanstd([dist1, dist2])
    return res
