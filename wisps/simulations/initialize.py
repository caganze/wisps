import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..data_analysis.initialize import OUTPUT_FILES
import splat
import wisps

splat.initializeStandards()

AREA=((4.1*385.+626.1)*(u.arcmin**2)).to((u.radian)**2) 

#add wisp pointings and take 2d-hst pointings to be ~600 arcmin***2 total
SOLID_ANGLE=(np.sin(np.sqrt(AREA)))**2.0
