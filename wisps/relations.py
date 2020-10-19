import pandas as pd 
import wisps

POLYNOMIAL_RELATIONS= pd.read_pickle(wisps.OUTPUT_FILES+'/polynomial_relations.pkl')
MAG_LIMITS=pd.read_pickle(wisps.OUTPUT_FILES+'/magnitude_cutoff.pkl.gz')
#from . import *