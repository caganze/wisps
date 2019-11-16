import pandas as pd 
import wisps

POLYNOMIAL_RELATIONS= pd.read_pickle(wisps.OUTPUT_FILES+'/polynomial_relations.pkl')
LUMINOSITY_FUCTION=pd.read_pickle(wisps.OUTPUT_FILES+'/luminosity_function.pkl')
#DES_LUMINOSITY_FUCTION=pd.read_pickle(wisps.OUTPUT_FILES+'/des_luminosity_function.pkl')
MAG_LIMITS=pd.read_pickle(wisps.OUTPUT_FILES+'/magnitude_cutoff.pkl')
OBSERVED_POINTINGS=pd.read_pickle(wisps.OUTPUT_FILES+'/observed_pointings.pkl')
#from . import *