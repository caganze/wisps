import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm
import splat
import wisps
import os
splat.initializeStandards()

SELECTION_FUNCTION=pd.read_pickle(wisps.OUTPUT_FILES+'/selection_function_lookup_table.pkl.gz')
SPGRID=np.arange(17, 42)
KIRK19_LF=pd.read_csv(wisps.WISP_PATH.split('/wisps')[0]+'/wisps/data/kirkpatricklf.txt')
MAG_LIMITS=pd.read_pickle(wisps.OUTPUT_FILES+'/magnitude_cutoff.pkl.gz')
WISP_PATH=os.environ['WISP_CODE_PATH']
EVOL_MODELS_FOLDER=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'/evmodels//')
