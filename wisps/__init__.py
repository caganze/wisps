from __future__ import print_function, division

__all__ = ["data_analysis", "simulations"]

#from .simulations import *
#from .simulations.initialize import *
#from .utils import *
from wisps.data_analysis import *
from wisps.data_analysis.initialize import *
from wisps.data_sets import datasets
from wisps.relations import *
from wisps.utils import *


import matplotlib as mpl 

##giving me a hard time

import seaborn
seaborn.set_style("ticks")


#matplotlib defaults
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['xtick.bottom']=True
mpl.rcParams['xtick.top']=True
mpl.rcParams['xtick.major.width']=0.9
mpl.rcParams['xtick.minor.width']=0.9
mpl.rcParams['ytick.major.width']=0.9
mpl.rcParams['ytick.minor.width']=0.9
mpl.rcParams['ytick.right']=True
mpl.rcParams['ytick.left']=True
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'



mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18) 
font = {'family' : 'Helvetica',
        'serif':[],
        'weight' : 'bold',
        'size'   : 18}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)
mpl.rcParams['agg.path.chunksize'] = 10000