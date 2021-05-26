#simple script that recalculates spectra but does it on the splat computer instead for speed 

import wisps
import pandas as pd 
def dumb_function(x):
	s=wisps.Source(filename=x)
	return s
#resave spectra 
#cands=#read in the data
cands=pd.read_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')
cands['spectra']=wisps.get_multiple_sources(cands.grism_id.replace('g141', 'G141').values)
cands.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')

#df_missing=pd.read_pickle(wisps.LIBRARIES+'/candidates_ids.pkl')

#missing=df_missing.grism_id.replace('g141', 'G141').values

#missing_spectra=wisps.get_multiple_sources(missing)

#missing_df['grism_id']=missing
#missing_df['spectra']=missing_spectra

#missing_df.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')


#df=pd.read_pickle(wisps.LIBRARIES+'/candidates_ids.pkl')
#df['grism_id']=df.grism_id.apply(lambda x: x.replace('g141', 'G141')).values
#print (df)
#df['spectra']=wisps.get_multiple_sources(df.grism_id.values)
#df.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')
