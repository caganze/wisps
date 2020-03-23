

from wisps.data_analysis.initialize import  *

datasets=dict()
#datasets['spex']=pd.read_pickle(LIBRARIES+'/spex_data_set_table.pkl')
datasets['stars']=COMBINED_PHOTO_SPECTRO_DATA
datasets['highsnr']=pd.read_pickle(LIBRARIES+'/highsnr_obejcts.pkl')
datasets['lowsnr']=pd.read_pickle(LIBRARIES+'/lowsnr_obejcts.pkl')
datasets['candidates']=pd.read_pickle(LIBRARIES+'/candidates.pkl')
datasets['traing_set']=pd.read_pickle(LIBRARIES+'/training_set.pkl')
#datasets['manjavacas']=pd.read_pickle(LIBRARIES+'/manjavacas.pkl')
#datasets['schneider']=pd.read_pickle(LIBRARIES+'/schneider.pkl')
datasets['rf_classified']=pd.read_pickle(LIBRARIES+'/labelled_by_rf.pkl')
datasets['rf_classified_not_indices']=pd.read_pickle(LIBRARIES+'/cands_not_indices.pkl')
datasets['ucd_data']=pd.read_pickle(OUTPUT_FILES+'/ucd_spectra.pkl')