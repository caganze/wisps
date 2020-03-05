#my colormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import splat
import pandas as pd
import splat.empirical as spem
import statsmodels.nonparametric.kernel_density as kde
import numba
from astropy.io import ascii
import matplotlib
from astropy import stats as astrostats
from scipy import stats
import bisect

#from . import *

#################
splat.initializeStandards()
###############
from wisps.utils import memoize_func


class Annotator(object):
    """
    Contains static method to manipulate index-index tables 
    """
    @staticmethod
    def  group_by_spt(df, **kwargs):
        
        """
        This is a static method that takes a table and an array of spectral type and 
        
        Args:
            df (pandas dataframe): a table of objects with a column of labelled spectral types

        Returns:
            returns the same table with spectral type ranges labelled

        spt_label=keyword for spt column
        """
        spt=kwargs.get('spt_label', 'Spts')
        #select by specral type range start spt=15
        df['spt_range']=''
        classes=['trash', 'M7-L0', 'L0-L5', 'L5-T0','T0-T5','T5-T9']
        if kwargs.get('assign_middle', False):
            #assign the the range to the median spectral type
            classes=[20, 22, 27, 32, 37]

        if kwargs.get('assign_number', False):
            classes=[0, 1, 2, 3, 4, 5]
        if not 'data_type' in df.columns:
            df['data_type']='templates'

        df['spt_range'].loc[(df[spt] < 17.0 ) & (df['data_type']== 'templates')]=classes[0]
        df['spt_range'].loc[(df[spt] >= 17.0 ) & (df[spt] <=20.0) & (df['data_type']== 'templates')]=classes[1]
        df['spt_range'].loc[(df[spt] >= 20.1 ) & (df[spt] <=25.0) & (df['data_type']== 'templates')]=classes[2]
        df['spt_range'].loc[(df[spt] >= 25.1 ) & (df[spt] <=30.0) & (df['data_type']== 'templates')]=classes[3]
        df['spt_range'].loc[(df[spt] >= 30.1 ) & (df[spt] <=35.0) & (df['data_type']== 'templates')]=classes[4]
        df['spt_range'].loc[(df[spt] >= 35.1 ) & (df[spt] <=40.0) & (df['data_type']== 'templates')]=classes[5]
        
        df['spt_range'].loc[ (df['data_type']== 'subdwarf')]='subdwarf'
        
        #print (df)
        if kwargs.get('add_subdwarfs', False):
            sds=kwargs.get('subdwarfs', None)
            #print ('adding subdwarfs')
            sds['spt_range']='subdwarfs'
            df=pd.concat([df,sds],  ignore_index=True, join="inner")
        #print (df)
        return df

    @staticmethod
    def color_from_spts(spts, **kwargs):
        """
        Given spt (or a bunch of intergers, get colors
        spts must be arrays of numbers else, will try to change it to colors
        """
        if isinstance(spts[0], str):
            try:
                spts=[float(x) for x in spts]
            except:
                spts=[splat.typeToNum(x) for x in spts]
                
        cmap=kwargs.get('cmap', matplotlib.cm.YlOrBr)
        maxi= np.nanmax(spts)
        mini=np.nanmin(spts)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors=[]
        for c in spts:
                colors.append(mapper.to_rgba(c))
        
        return colors

    @staticmethod
    def reformat_table(df):
        """
        drop uncertainties in the indidces 
        """
        new_df=pd.DataFrame()
        for k in df.columns: 
            if isinstance(df[k].iloc[0], tuple):
                new_df[k]=np.array(np.apply_along_axis(list, 0, df[k].values))[:,0]
            else:
                new_df[k]=df[k].values
        return new_df

@np.vectorize      
def splat_teff_to_spt(teff):
    rel=splat.SPT_TEFF_RELATIONS['pecaut']
    spt_sorted_idx=np.argsort(rel['values'])
    scatter=108
    teffsc=np.random.normal(teff, scatter)
    return np.interp(teffsc, np.array(rel['values'])[spt_sorted_idx], np.array(rel['spt'])[spt_sorted_idx])

@numba.jit
def make_spt_number(spt):
    ##make a spt a number
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

#pecaut constants
mamjk=ascii.read('/users/caganze/research/wisps/data/mamajek_relations.txt').to_pandas().replace('None', np.nan)
pec_js=mamjk.M_J.apply(float).values
pec_jminush=mamjk['J-H'].apply(float).values
pec_hs=pec_js-pec_jminush
pec_spts=mamjk.SpT.apply(make_spt_number).apply(float).values
pec_hsortedindex=np.argsort(pec_hs)
pec_jsortedindex=np.argsort(pec_js)

def absolute_magnitude_jh(spt):
    """
    returns J and H magnitudes by interpolating between values from pecaut2013
    must probably sort spt if spt is a list bfeore passing it through the interpolator
    """    
    hval=np.interp(spt,  pec_spts[pec_hsortedindex], pec_hs[pec_hsortedindex])
    jval=np.interp(spt,  pec_spts[pec_jsortedindex], pec_js[pec_jsortedindex])
    
    return [jval, hval]


@numba.vectorize("float64(float64, float64)", target='cpu')
def get_distance(absmag, rel_mag):
    return 10.**(-(absmag-rel_mag)/5. + 1.)

@numba.jit
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

        
def drop_nan(x):
    x=np.array(x)
    return x[(~np.isnan(x)) & (~np.isinf(x)) ]


def custom_histogram(things, grid, binsize=1):
    n=[]
    for g in grid:
        n.append(len(things[np.logical_and(g<=things, things< g+binsize)]))
    return np.array(n)



@numba.jit
def is_in_that_classification(spt, subclass):
    #determines if a spt is within a subclass
    flag=False
    scl=subclass.lower()
    if scl[0] in ['m', 'l', 't']:
        slow=splat.typeToNum(subclass[:2])
        shigh=splat.typeToNum(subclass[-2:])
        if slow<=make_spt_number(spt)<=shigh:
            flag=True
    if scl.startswith('y') & (make_spt_number(spt)>=38):
        flag=True
    if scl.startswith('subd'):
        flag=False
    
    return flag



def random_draw(xvals, cdfvals, nsample=10):
    """
    randomly drawing from a discrete distribution
    """
    @numba.vectorize("int32(float64)")
    def invert_cdf(i):
        return bisect.bisect(cdfvals, i)-1
    x=np.random.rand(nsample)
    idx=invert_cdf(x)
    return np.array(xvals)[idx]


def fit_polynomial(x, y, n=2, y_unc=None, sigma_clip=False, sigma=None):
    """
    Polynomial fit with n-sigma clipping
    """
    if sigma_clip:
        va=np.array([x, y]).T
        d=pd.DataFrame(va).dropna().values
        sigma_clipped=astrostats.sigma_clip(d, sigma=sigma)
        x=sigma_clipped[:,0]
        y=sigma_clipped[:,1]

    nany=np.isnan(x)
    p = np.poly1d(np.polyfit(x[~nany], y[~nany], n))

    if y_unc is not None:
        p=np.poly1d(np.polyfit(x[~nany],y[~nany], n, w=1./y_unc[~nany]))
    return p

def kernel_density(distr):
    """
    1D-kernel density estimation
    """
    kernel = stats.gaussian_kde(distr)
    return kernel


def dropnans(x):
    return [~np.isnan(x)]