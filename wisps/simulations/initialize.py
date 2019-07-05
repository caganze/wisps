import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..data_analysis.initialize import OUTPUT_FILES
from astropy.coordinates import SkyCoord
import splat
import wisps

splat.initializeStandards()

AREA=(4.3*(u.arcmin**2)).to((u.radian)**2)
SOLID_ANGLE=(np.sin(np.sqrt(AREA)))**2.0

POLYNOMIAL_RELATIONS= pd.read_pickle(wisps.OUTPUT_FILES+'/polynomial_relations.pkl')
LUMINOSITY_FUCTION=pd.read_pickle(OUTPUT_FILES+'/luminosity_function.pkl')
obs=pd.read_csv(wisps.OUTPUT_FILES+'//observation_log_with_limit.csv')
OBSERVED_POINTINGS= SkyCoord(obs['ra (deg)'].values, obs['dec(deg)'].values, unit=u.deg)