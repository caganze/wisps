#save dictionary 
import pandas as pd
import pickle

folder='~/research/wisps/db//'
ds=pd.read_pickle(folder+'/distance_samples.pkl')

for k in ds.keys():
	with open(folder+'distance_samples{}'.format(k), 'wb') as pfile:
    	pickle.dump(ds[k], pfile, protocol=pickle.HIGHEST_PROTOCOL)
