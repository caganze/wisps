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
import os
#from . import *

#################
splat.initializeStandards()
###############

from wisps.utils import memoize_func
WISP_PATH=os.environ['WISP_CODE_PATH']
DATA_FILES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//data//')
LIBRARIES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//libraries//')
kirkpa2019pol={'pol':np.poly1d(np.flip([36.9714, -8.66856, 1.05122 ,-0.0344809])), 
                    'scatter':.67, 'range':[36, 44]}

def plot_annotated_heatmap(ax, data, gridpoints, columns, cmap='viridis', 
                           annotate=False, vmin=0.0, vmax=1.0, textsize=14):
    #plot an annotated heatmap
    data= data.dropna()
    xcol, ycol, zcol= columns
    step1= np.ptp(data[xcol])/gridpoints
    step2= np.ptp(data[ycol])/gridpoints
    
    #print (step1)
    
    xgrid= np.linspace(data[xcol].min(), data[xcol].max(), gridpoints)
    ygrid= np.linspace(data[ycol].min(), data[ycol].max(), gridpoints)
    
    
    mask = np.zeros((len(xgrid), len(ygrid)))
    values = np.zeros((len(xgrid), len(ygrid)))
    #for annotation
    for i in range(len(xgrid)):
        #loop over matrix
        for j in range(len(ygrid)):
            if (i == len(xgrid)-1) | (j == len(ygrid)-1) :
                pass
            else:
                maskx= np.logical_and(data[xcol] > xgrid[i], data[xcol] <= xgrid[i]+step1)
                masky=np.logical_and(data[ycol] > ygrid[j], data[ycol] <=ygrid[j]+step2)
                zmedian= np.nanmean(data[zcol][np.logical_and(maskx, masky)])
                lenz= len(data[np.logical_and.reduce([maskx, masky])])

                if lenz == 0:
                    values[j][i] = np.nan
                    mask[j][i] = 1
                else:
                    values[j][i] = zmedian
                    if annotate == 'third_value':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{zmedian:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                    if annotate== 'number':
                        ax.text(xgrid[i]+step1/2., ygrid[j]+step2/2., f'{lenz:.0f}',
                                 ha='center', va='center', fontsize=textsize, color='#111111')
                
    values2 = np.ma.array(values, mask=mask)
    cax = ax.pcolormesh(xgrid, ygrid, values2, vmin=vmin, vmax=vmax, cmap=cmap)
    #plt.axis('tight')
    ymin, ymax = plt.ylim()

    ax.minorticks_on()

    ax.set_ylim(ymax, ymin)
    return 

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
                new_df[k+'_er']=np.array(np.apply_along_axis(list, 0, df[k].values))[:,1]
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

    
@np.vectorize      
def splat_teff_from_spt(spt):
    rel=splat.SPT_TEFF_RELATIONS['pecaut']
    #spt_sorted_idx=np.argsort(rel['values'])

    teff=np.interp(spt, np.array(rel['spt']),  np.array(rel['values']))
    return np.random.normal(teff, 108)

    
    
@numba.jit
def make_spt_number(spt):
    ##make a spt a number
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt

#def get_abs_mag_contants():
#need to wrap these into a function to avoid overloading memory
mamjk=ascii.read(DATA_FILES+'/mamajek_relations.txt').to_pandas().replace('None', np.nan)
pec_js=mamjk.M_J.apply(float).values
pec_jminush=mamjk['J-H'].apply(float).values
pec_hs=pec_js-pec_jminush
pec_spts=mamjk.SpT.apply(make_spt_number).apply(float).values
pec_hsortedindex=np.argsort(pec_hs)
pec_jsortedindex=np.argsort(pec_js)


best_dict={'2MASS J': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [10.36,10.77,11.15,11.46,11.76,12.03,12.32,12.77,13.51,13.69,14.18,14.94,14.90,14.46,14.56,15.25,14.54,14.26,13.89,14.94,15.53,16.78,17.18,17.75],\
            'rms': [0.30,0.30,0.42,0.34,0.18,0.15,0.21,0.24,0.28,0.25,0.60,0.20,0.13,0.71,0.5,0.12,0.06,0.16,0.36,0.12,0.27,0.76,0.51,0.5]},
        '2MASS H': {\
            'spt': [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39], \
            'values': [9.76,10.14,10.47,10.74,11.00,11.23,11.41,11.82,12.45,12.63,13.19,13.82,13.77,13.39,13.62,14.39,13.73,13.67,13.57,14.76,15.48,16.70,17.09,17.51],\
            'rms': [0.30,0.31,0.43,0.35,0.23,0.21,0.25,0.29,0.3,0.30,0.62,0.31,0.20,0.73,0.5,0.18,0.15,0.24,0.40,0.24,0.37,0.78,0.5,0.5]}}

def absolute_mag_kirkpatrick(spt, filt):
    if filt != '2MASS H':
        return np.nan
    else:
        if (spt > 36) and (spt <44):
            pol=kirkpa2019pol['pol']
            unc=kirkpa2019pol['scatter']
            return np.random.normal(pol(spt-30), unc, 1000).mean()
            
        else:
            return np.nan

@np.vectorize
def absolute_magnitude_jh(spt):
    """
    returns J and H magnitudes by interpolating between values from pecaut2013
    must probably sort spt if spt is a list bfeore passing it through the interpolator
    """ 
    jval, hval=(np.nan, np.nan)
    #[SHOULD ADD VEGA TO AB CONVERSION FACTOR]
    if spt <=37:
        hval=np.interp(spt,  pec_spts[pec_hsortedindex], pec_hs[pec_hsortedindex])
        jval=np.interp(spt,  pec_spts[pec_jsortedindex], pec_js[pec_jsortedindex])
    else:
        hval=absolute_mag_kirkpatrick(spt, '2MASS H')

    return jval, hval

def k_clip_fit(x, y, sigma_y, sigma = 5, n=6):
    
    '''Fit a polynomial to y vs. x, and k-sigma clip until convergence
    hard-coded, returns mask array
    '''
    
    not_clipped = np.ones_like(y).astype(bool)
    n_remove = 1
    
    #use median sigma
    #median_sigma= np.nanmedian(sigma_y)
    
    while n_remove > 0:

        best_fit = np.poly1d(np.polyfit(x[not_clipped], y[not_clipped], n))
        
        norm_res = (np.abs(y - best_fit(x)))/(sigma_y)
        remove = np.logical_and(norm_res >= sigma, not_clipped == 1)
        n_remove = sum(remove)
        not_clipped[remove] = 0   
        
    return  not_clipped

def fit_with_nsigma_clipping(x, y, y_unc, n, sigma=3.):
    not_clipped = k_clip_fit(x, y, y_unc, sigma = sigma)
    return not_clipped, np.poly1d(np.polyfit(x[not_clipped], y[not_clipped], n))


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

def kernel_density(distr, **kwargs):
    """
    1D-kernel density estimation
    """
    kernel = stats.gaussian_kde(distr, **kwargs)
    return kernel



def get_big_file():
    COMBINED_PHOTO_SPECTRO_FILE=LIBRARIES+'/master_dataset.h5'
    COMBINED_PHOTO_SPECTRO_DATA=pd.read_hdf(COMBINED_PHOTO_SPECTRO_FILE, key='new')
    #definitions
    return COMBINED_PHOTO_SPECTRO_DATA