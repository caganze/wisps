# -*- coding: utf-8 -*-

##giving me a hard time
try: 
    import seaborn
    seaborn.set_style("ticks")
except:
	print ('could not import seaborn')
	pass
from astropy import units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import warnings


from ..utils.tools import MYCOLORMAP


#matplotlib defaults
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.2
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'


#environment variables
if os.environ.get('READTHEDOCS') == 'True' : 
	pass
	
else:
    WISP_PATH=os.environ['WISP_CODE_PATH']
    REMOTE_FOLDER=os.environ['WISP_SURVEY_DATA']
    INDICES=os.environ['WISP_INDICES_DATA']
    REMOTE_PHOT_CATALOGS=INDICES.split('Indices')[0]+'/catalogs//'
    #units
    F_UNITS=u.erg/(u.cm**2 * u.s * u.micron) #default units for wavelength and flux
    W_UNITS=u.micron
    #output files
    OUTPUT_FILES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'/db//')
    OUTPUT_FIGURES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//figures//')
    OUTPUT_TABLES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//tables//')
    LIBRARIES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//libraries//')
    #read in the photometry
    #PHOTOMETRY_FILE= OUTPUT_FILES+'/combined_wisp_hst3_photometry.pkl'
    #PHOTOMETRY_DATA=pd.read_pickle(PHOTOMETRY_FILE)
    COMBINED_PHOTO_SPECTRO_FILE=LIBRARIES+'/combined_phot_spec_all.hdf'
    COMBINED_PHOTO_SPECTRO_DATA=pd.read_hdf(COMBINED_PHOTO_SPECTRO_FILE, 'all_phot_spec_data')
    #definitions
    INDEX_NAMES=np.array(['H_2O-1/J-Cont',  'H_2O-2/H_2O-1', 'H-cont/H_2O-1', 'CH_4/H_2O-1',   'H_2O-2/J-Cont',   'H-cont/J-Cont', 'CH_4/J-Cont',    
                'H-cont/H_2O-2',         'CH_4/H_2O-2',  'CH_4/H-Cont'] ) 

