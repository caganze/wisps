#add the number of degrees of freedom to the master table
import pandas as pd 
import numpy as np


def reformat_wisp_grism_id(grism_id):
	
    if grism_id.endswith('.ascii'):
        return grism_id.split('ASCII/')[-1].split('.1D.ascii')[0].lower()
    if grism_id.endswith('.dat'):
        return grism_id.split('lsp_wisp_hst_wfc3_')[-1].split('a_')[0].lower()

alldata=pd.read_hdf('~/research/wisps//libraries/master_dataset.hdf',  key='new')
df=pd.read_hdf('~/research/wisps//libraries/all_wisp_spectrasept162020.h5', key='hst3d')
other_keys=np.append(['wisp'], ['wisp{}'.format(x) for x in np.arange(2, 10)])
dfwisp=pd.concat(pd.read_hdf('~/research/wisps//libraries/all_wisp_spectrasept162020.h5', key=k) for k in other_keys )
dfwisp=dfwisp[dfwisp.grism_id.str.contains('g141')]
dfnjoined=pd.concat([dfwisp, df])
dfnjoined['grism_id']=dfnjoined.grism_id.apply(reformat_wisp_grism_id)
merged=alldata.merge(dfnjoined, how='inner', on='grism_id')

print (merged.shape)