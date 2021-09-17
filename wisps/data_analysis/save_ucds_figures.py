import wisps
import pandas as pd
import splat
import numpy as np
ucds=pd.read_pickle(wisps.LIBRARIES+'/new_real_ucds.pkl')
print (len(ucds))

#ucds['f_test']=ucds.spectra.apply(lambda x: x.f_test)
#ucds['dof']=ucds.spectra.apply(lambda x: x.dof)
#ucds['line_chi']=ucds.spectra.apply(lambda x: x.line_chi)
#ucds['spex_chi']=ucds.spectra.apply(lambda x: x.spex_chi)
#ucds['wavenumber']=ucds.spectra.apply(lambda x:len(x.wave))

fold='/Users/caganze/research/wisps/figures/ltwarfs/'

ids=0
def reclassify(s):
    if s is None:
        return None
    #spt_unc=s.spectral_type[1
    #rngs=[[1.17,1.35],[1.45,1.67]]
    rngs=[[1.15, 1.65]]
    spt, spt_e= splat.classifyByStandard(s.splat_spectrum, fitrange=[[1.15, 1.65]], 
                                         sptrange=['M0','Y1'], average=True)
    #s.classify_by_standard(comprange=rngs)
    #print(wisps.make_spt_number(spt))
    #spt, spt_e=wisps.classify_by_templates(s, comprange=rngs)
    s.spectral_type=(np.round(wisps.make_spt_number(spt)), spt_e)
    #s.calculate_distance(use_spt_unc=True, use_index_type=False)
    return s

for idx, row in ucds.iterrows():
    try:
    	#print ()
        s=row.spectra
        filename=fold+'spectrum'+str(idx)+'.pdf'
        s.pixels_per_image=100
        #DON'T FIT SD TO T TYPES
        if  wisps.make_spt_number(s.spectral_type[0])<30:
            s.plot(compare_to_sds=True, comprange=[[1.15, 1.6]], save=True, filename=filename, dpi=160)
        else:
            s.plot(compare_to_sds=False, comprange=[[1.15, 1.6]], save=True, filename=filename, dpi=160)

    except:
        s=wisps.Source(filename=row.grism_id.replace('g141', 'G141'),is_ucd=False)
        s=reclassify(s)
        print (filename)
        #print (s.spectral_type)
        #s.spectral_type=(row.spt, row.spt_er)
        s.pixels_per_image=100
        #filename=fold+'spectrum'+str(s.designation)+'.pdf'
        filename=fold+'spectrum'+str(idx)+'.pdf'
        if  wisps.make_spt_number(s.spectral_type[0])<30:
            s.plot(compare_to_sds=True, comprange=[[1.15, 1.6]], save=True, filename=filename, dpi=160)
        else:
            s.plot(compare_to_sds=False, comprange=[[1.15, 1.6]], save=True, filename=filename, dpi=160)