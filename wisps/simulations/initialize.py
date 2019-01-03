import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..data_analysis.initialize import OUTPUT_FILES
import splat
splat.initializeStandards()

CONSTANTS={'RSUN': 8.3*u.kpc.to(u.pc)*u.pc,
			'ZSUN': 2.7*u.kpc.to(u.pc)*u.pc,
			'N_0':0.39*u.pc**-3,
			'CCD_SIZE':1*u.arcmin**2,
			'SATURATION_LIMIT': {'J': 5}}



LUMINOSITY_FUCTION=pd.read_pickle(OUTPUT_FILES+'/luminosity_function.pkl')