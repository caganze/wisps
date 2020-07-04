#simple script that recalculates spectra but does it on the splat computer instead for speed 

import wisps
import pandas as pd 
def dumb_function(x):
	print (x)
	s=wisps.Source(filename=x)
	return s

df=pd.read_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='good')
print (df)
df['grism_id']=df.grism_id.apply(lambda x: x.replace('g141', 'G141')).values
df['spectra']=wisps.get_multiple_sources(df.grism_id.values)
df.to_hdf(wisps.LIBRARIES+'/objects_of_interest.hdf', key='good')
