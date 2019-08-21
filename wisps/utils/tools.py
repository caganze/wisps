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

@numba.jit
def make_spt_number(spt):
    ##make a spt a number
    if isinstance(spt, str):
        return splat.typeToNum(spt)
    else:
        return spt