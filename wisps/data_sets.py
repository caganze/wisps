

from wisps.data_analysis.initialize import  *


datasets=dict()
datasets['spex_data_set']=pd.read_pickle(LIBRARIES+'/spex_data_set.pkl')
datasets['aegis_cosmos']=pd.read_pickle(LIBRARIES+'/aegis_cosmos.pkl')
datasets['combined']=COMBINED_PHOTO_SPECTRO_DATA