

from wisps.data_analysis.initialize import  *


datasets=dict()
datasets['spex']=pd.read_pickle(LIBRARIES+'/spex_data_set.pkl')
datasets['all']=COMBINED_PHOTO_SPECTRO_DATA
datasets['candidates']=pd.read_csv(LIBRARIES+'/candidates.csv')
datasets['traing_set']=pd.read_pickle(LIBRARIES+'/training_set.pkl')
datasets['manjavacas']=pd.read_pickle(LIBRARIES+'/manjavacas.pkl')
datasets['schneider']=pd.read_pickle(LIBRARIES+'/schneider.pkl')