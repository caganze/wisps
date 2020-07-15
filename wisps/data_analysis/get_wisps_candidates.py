#simple script that recalculates spectra but does it on the splat computer instead for speed 

import wisps
import pandas as pd 
def dumb_function(x):
	print (x)
	s=wisps.Source(filename=x)
	return s

df=pd.read_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')

#df_missing=pd.read_pickle(wisps.LIBRARIES+'/candidates_ids.pkl')


missing=df.grism_id[df.spectra.isna()].values
print (missing)
missing_spectra=wisps.get_multiple_sources(missing)

missing_df=pd.DataFrame()
missing_df['grism_id']=missing
missing_df['spectra']=missing_spectra

dfn=pd.concat([df, df_missing])
dfn.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')


#df=pd.read_pickle(wisps.LIBRARIES+'/candidates_ids.pkl')
#df['grism_id']=df.grism_id.apply(lambda x: x.replace('g141', 'G141')).values
#print (df)
#df['spectra']=wisps.get_multiple_sources(df.grism_id.values)
#df.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='all')
