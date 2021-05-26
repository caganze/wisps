import wisps
import pandas as pd
import splat
import numpy as np

ucds=pd.read_pickle(wisps.LIBRARIES+'/new_real_ucds.pkl')
ucds['f_test']=ucds.spectra.apply(lambda x: x.f_test)
ucds['dof']=ucds.spectra.apply(lambda x: x.dof)
ucds['line_chi']=ucds.spectra.apply(lambda x: x.line_chi)
ucds['spex_chi']=ucds.spectra.apply(lambda x: x.spex_chi)
ucds['wavenumber']=ucds.spectra.apply(lambda x:len(x.wave))

fold='/Users/caganze/research/wisps/figures/ltwarfs/'

ids=0
for idx, row in ucds.iterrows():
    try:
    	#print ()
        s=row.spectra
        filename=fold+'spectrum'+str(ids)+'.pdf'
        print (filename)
        s.pixels_per_image=100
        s.plot(save=True, filename=filename)
        ids=ids+1
    except:
        s=wisps.Source(filename=row.grism_id.replace('g141', 'G141'),is_ucd=False )
        print (filename)
        s.pixels_per_image=100
        filename=fold+'spectrum'+str(ids)+'.pdf'
        s.plot(save=True, filename=filename)
        ids=ids+1