# -*- coding: utf-8 -*-


from astropy import units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import warnings


from ..utils.tools import *


import seaborn
seaborn.set_style("ticks")


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
    #OUTPUT_FILES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'/db//')
    OUTPUT_FIGURES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//figures//')
    OUTPUT_TABLES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//tables//')
    #LIBRARIES=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//libraries//')
    LIBRARIES='/volumes/TOSHIBA/wispsdata/libraries/'
    OUTPUT_FILES='/volumes/TOSHIBA/wispsdata/db/'
    #read in the photometry
    #PHOTOMETRY_FILE= OUTPUT_FILES+'/combined_wisp_hst3_photometry.pkl'
    #PHOTOMETRY_DATA=pd.read_pickle(PHOTOMETRY_FILE)
    INDEX_NAMES=np.array(['H_2O-1/J-Cont',  'H_2O-2/H_2O-1', 'H-cont/H_2O-1', 'CH_4/H_2O-1',   'H_2O-2/J-Cont',   'H-cont/J-Cont', 'CH_4/J-Cont',    
                'H-cont/H_2O-2',         'CH_4/H_2O-2',  'CH_4/H-Cont', 'H_2O-1+H_2O-2/J-Cont', 'H_2O-1+H_2O-2/H-Cont', 'H_2O-1+CH_4/J-Cont',
                'H_2O-2+CH_4/J-Cont',  'H_2O-1+CH_4/H-Cont', 'H_2O-2+CH_4/H-Cont'] ) 
    
    #colorscheme for filters 
    FILTER_COLOR_SCHEME={'F110W':'#0074D9' , 'F140W':'#FF851B', 'F160W':'#FF4136'}


