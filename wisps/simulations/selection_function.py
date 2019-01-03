###########
#simulating the selection function
##########


from .initialize import pd, np
from tqdm import tqdm
import copy
import wisps
#from .initialize import datasets

def simulate_mags(sp, mag0, nsample=100, sigma=0.1):
    """
    
    simulate a range of spectra starting with a given magnitude
    return their snr, mag distributions
    """
    #copy the object to modify it without changing it
    spcopy=copy.deepcopy(sp)
    #mask between ranges that I care about 
    mask=np.where((sp.wave>1.15) &(sp.wave<1.65))[0]
    #measure the snr of the spectrum
    old_flux=np.nanmedian(sp.flux)
    mags=[]
    snrs=[]
    sps=[]
    ftests=[]
    for i in tqdm(np.arange(nsample)):
                #if (mag0<30) or (i < nsample):  dont' worry about reaching magnitude 30 yet
                ns=sigma*np.random.normal(np.nanmean(sp.noise[mask]),np.nanstd(sp.noise[mask]),len(sp.flux))
                spcopy.add_noise(noise=ns)
                new_flux=np.nanmedian(spcopy.flux)
                mag=mag0-2.5*np.log10(new_flux/old_flux)
                mags.append(mag)
                snrs.append(spcopy.snr['snr2'])
                sps.append(spcopy)
                ftests.append(spcopy.f_test()['f'])

    return pd.DataFrame({'mag':np.array(mags), 'snr':np.array(snrs), 'spectra':sps, 'f':np.array(ftests)})

def use_criteria(crt, df):
        ##lazy way to avoid a for loop
        #use selection criteria 
        #actually i didn't even avoid the loop oh well
        f=0.0
        if len(df)==0:
            f=0.0
        else:
            if not isinstance(crt, list):
                cols=np.append(crt.name.split(), 'Names')
                f=len(np.unique(np.concatenate(crt.select(df=df[cols]))))/len(df)
            else:
                slct=[]
                for c in crt:
                     cols=np.append(c.name.split(), 'Names')
                     slct.append(np.unique(np.concatenate(c.select(df=df[cols]))))
                f=len(np.unique(np.concatenate(slct)))/len(df)

        return f

def selection_function(sp, mag, selection, nsample=100.0, sigma=0.1):
    """
    Return the fraction of spectra selected per magnitude bin per spectral type
    sp: the spectrum object to add noise to
    spt: its spectral type
    mag: the magnitude of the object
    selection; dictionary with keywords, snr: snrcutoff, f: f-test cutoff, crt selection to use
    nsmaple: how many spectra to simulate
    """

    res=simulate_mags(sp, float(mag), nsample=nsample, sigma=0.1)
    indices=pd.DataFrame([x.indices for x in res.spectra])
    res['Names']=['sp'+str(i) for i in np.arange(len(res))]
    for k in indices.columns: res[k]=indices[k]
    res=wisps.Annotator.reformat_table(res)
    bins=np.array_split(res, 100)
    f_selected=[]
    mean_mag=[]
    crt=selection['crt']
    for b in bins:
        #print (b)
        b1=b[(b.snr>selection['snr']) & (b.f>selection['f'])]
        f_selected.append(use_criteria(crt, b1))
        mean_mag.append(np.nanmedian(b.mag))
        #turn magntitude into distance
    return mean_mag, f_selected