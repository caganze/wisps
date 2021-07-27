#!/usr/bin/env python
# coding: utf-8

# In[1]:


import wisps
import numpy as np
import matplotlib.pyplot as plt
import wisps.simulations as wispsim
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import Normalize
import astropy.units as u 
import wisps.simulations.effective_numbers as eff
import seaborn as sns
import matplotlib
import popsims
import itertools
#plt.style.use('dark_background')


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wispsim.MAG_LIMITS


# In[3]:


import popsims
import splat


# In[4]:


sgrid=wispsim.SPGRID
pnts=pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')
corr_pols=wisps.POLYNOMIAL_RELATIONS['mag_limit_corrections'] 
klf=pd.read_csv('/users/caganze/research/wisps/data/kirkpatricklf.txt', delimiter=',')
klf['bin_center']=np.mean(np.array([klf.t0.values, klf.tf.values]), axis=0)
klf=klf.replace(0.0,np.nan)

ucds=pd.read_pickle(wisps.LIBRARIES+'/new_real_ucds.pkl')
#cands=cands[(cands.spt >=17) & (cands.snr1>=3)].reset_index(drop=True)
cands=(ucds[ucds.selection!='']).reset_index(drop=True)
tab=wisps.Annotator.reformat_table(cands)
pnt_names=[x.name for x in pnts]


# In[5]:


#spgrid


# In[6]:


#cmap= sns.color_palette("coolwarm", 8, as_cmap=True)
cmap=matplotlib.cm.get_cmap('coolwarm')
cnorm=Normalize(wispsim.HS[0], (wispsim.HS[-1]))


# In[7]:


kirkpatrick2020LF={'bin_center':np.flip(np.array([2025, 1875, 1725, 1575, 1425, 1275, 1125 , 975, 825, 675, 525])), 
                   'values':np.flip(np.array([0.72, 0.50,0.78, 0.81,0.94, 1.95, 1.11, 1.72, 1.99, 2.80, 4.24])), 
                   'unc':np.flip(([0.18, 0.17, 0.20,0.20, 0.22, 0.3, 0.25, 0.3, 0.32, 0.37, 0.70]))}


# In[8]:


MODEL_NAMES=['burrows1997', 'burrows2001', 'baraffe2003', 'saumon2008', 'marley2019', 'phillips2020']
MODEL_SHORT_NAMES=['B97', 'B01', 'B03', 'SM08', 'M19', 'P20']


# In[9]:


def bin_by_spt_bin(sp_types, number, ltonly=False):
    ranges=[[17, 20], [20, 25], [25, 30], [30, 35], [35, 40]]
    if ltonly:
        ranges=[[17, 20], [20, 30], [30, 41]]
    numbers=[]
    for r in ranges:
        idx= np.logical_and((r[0]<=sp_types), (r[1]>sp_types))
        numbers.append(np.nansum(number[idx]))
    return numbers

def get_all_numbers():
    #Distribute the parameter sets evenly across the cores
    func=lambda x, y:  get_simulated_number_model(y, x)

    paramlist=[(i, j)  for i, j in itertools.product(MODEL_NAMES, wispsim.HS)]
    res  = [func(x, y) for x,y in tqdm(paramlist)]
    
    nbrs = {}
    for k in MODEL_NAMES:
        ds0={}
        for j in res:
            if k in j.keys():
                key=[x for x in j[k].keys()][0]
                ds0.update({key: [(j[k][key])[yi] for yi in wispsim.SPGRID]})
        #print (ds0)
        nbrs[k]=np.array([ds0[k] for k in wispsim.HS])

    return nbrs
    


def get_pointing(grism_id):
    if grism_id.startswith('par'):
        pntname=grism_id.lower().split('-')[0]
    else:
        pntname=grism_id.lower().split('-g141')[0]
    loc=pnt_names.index(pntname)
    return np.array(pnts)[loc]


def iswithin_mag_limits(mags, pnt, spt):
    #mgs is a dictionary
    flags=[]
    for k in pnt.mag_limits.keys():
        if k =='F110' and pnt.survey =='hst3d':
            flags.append(True)
        else:
            flags.append(mags[k] <= pnt.mag_limits[k]+ (corr_pols[k+'W'][0])(spt))
    return np.logical_or.reduce(flags)

def get_simulated_number_model(hidx, model):
    #hidx is a scale height, model is evolutionary model
    df=pd.read_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut_binaries.h5',                       key=str(model)+str(hidx)+str('spt_abs_mag'))
    cutdf=(df[~df.is_cut]).rename(columns={'temperature': 'teff',                                           'slprob': 'sl'})
    #cutdf=pd.read_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut.h5', key=str(model)+str('h')+str(hidx)+'F110_corrected')
    #scl_dict=pd.read_pickle(wisps.OUTPUT_FILES+'/lf_scales.pkl') 
    #scales=scl_dict[model]
    scale=[cutdf.scale.mean(), cutdf.scale_unc.mean(), cutdf.scale_times_model.mean()]
    #scale=scale_lf_teff(cutdf.teff)
    NSIM=dict(zip(wispsim.SPGRID,np.zeros((len(wispsim.SPGRID), 2))))
    cutdf['spt_r']=cutdf.spt.apply(np.round)
    for g in cutdf.groupby('spt_r'):
        sn= len(cutdf.teff[np.logical_and(cutdf.teff>=450, cutdf.teff<=2100)])
        n0=scale[-1]/scale[0]
        #print (n0)
        scln=np.array([scale[0]*n0/sn,                       (scale[1]*scale[-1])/(sn*scale[0])])
        #scln=np.array(scale)
        #assert scln[0] > scale[0]
        NSIM[g[0]]=np.nansum(g[1].sl)*scln
    del cutdf
    return {model: {hidx:NSIM}}


# In[10]:


MODEL_NAMES


# In[ ]:





# In[11]:


#cutdf.scale


# In[12]:


def plot_one(NUMBERS, VOLUMES, filename='/oberved_numbers_one_panel.pdf'):
    data_to_save={}
    # In[ ]:
    nall=wisps.custom_histogram(cands.spt.apply(wisps.make_spt_number), sgrid, 1)
    
    y2=bin_by_spt_bin(wispsim.SPGRID,nobs, ltonly=False)-THICK
    yall=bin_by_spt_bin(wispsim.SPGRID,nall, ltonly=False)-THICK
    
    dy2=np.sqrt(y2)
    dyall=np.sqrt(yall)
     #add this to the dictionary
    data_to_save['nall']=nall
    data_to_save['nobs']=nobs
    data_to_save['yall']=yall
    data_to_save['y2']=y2

    fig, a=plt.subplots(figsize=(8, 6))
    
    #for model, a in zip(['baraffe2003', 'saumon2008', 'marley2019', 'phillips2020'], np.concatenate(ax)):
    model='baraffe2003'
    for idx, h in enumerate(wispsim.HS):
            
            ns=None
            ns=((NUMBERS[model])[idx])[:,0]*VOLUMES[idx]
            nuncs=((NUMBERS[model])[idx])[:,1]*VOLUMES[idx]
            
            a.plot(spgrid2, bin_by_spt_bin(wispsim.SPGRID,ns, ltonly=False), 
                          color= cmap(cnorm(h)), 
                   linewidth=3, drawstyle="steps-mid")
            a.fill_between(spgrid2, bin_by_spt_bin(wispsim.SPGRID,ns+nuncs, ltonly=False),  
                           bin_by_spt_bin(wispsim.SPGRID,ns-nuncs, ltonly=False), alpha=0.5, 
                           color= cmap(cnorm(h/100)),  step="mid")
        
        
    a.set_yscale('log')
    #a.errorbar(spgrid2,y2, yerr=dy2,fmt='o', color='#111111')
    #a.errorbar(spgrid2,yall, yerr=dyall,color='#B10DC9', mfc='white', fmt='o')
    a.set_xlabel('SpT',fontsize=18)
    a.set_ylabel('N',fontsize=18)
    a.minorticks_on()
            

    #a.set_title('Model= SM08', fontsize=18)
    a.set_title('Model= B03', fontsize=18)
    #a.set_title('Model= M19', fontsize=18)
    #a.set_title('Model= P20', fontsize=18)

    a.errorbar(spgrid2,y2, yerr=dy2,fmt='o', label='Mag Limited')
    #a.errorbar(spgrid2,yall, yerr=dyall, fmt='o', label='All Observations')
    
    cax = fig.add_axes([.5, 0.7, .3, 0.03])
    mp=matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    cbar=plt.colorbar(mp, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel(r'Scaleheight (H)', fontsize=18)
    #cbar.ax.set_yticks([1, 3, 5, 10])
    #a.legend(fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.savefig(wisps.OUTPUT_FIGURES+filename, bbox_inches='tight')
    


# In[13]:


#d=pd.read_pickle(wisps.OUTPUT_FILES+'/distance_samples{}'.format(h))


# In[14]:


#expectted counts from thick disk
THICK=np.array([8.79798048, 2.30571423, 0.14145726, 0.08853498, 0.01784511])


# In[15]:



tab['pnt']=tab['grism_id'].apply(get_pointing)
tab['spt_val']=np.vstack(tab.spt.values)[:,0]
obsmgs=tab[['F140W', 'F110W', 'F160W']].rename(columns={"F110W": "F110", 
                                                                    "F140W": "F140",
                                                                    "F160W": "F160"}).to_dict('records')

flags=[iswithin_mag_limits(x, y, z) for x, y, z in zip(obsmgs, tab.pnt.values,tab.spt.values )]

#let's see what happens if we include all objects
#flags=np.ones(len(flags)).astype(bool)
cdf_to_use=tab[flags]

nobs=wisps.custom_histogram(cdf_to_use.spt_val.apply(wisps.make_spt_number), sgrid, 1)


spgrid2=['M7-L0', 'L0-L5', 'L5-T0', 'T0-T5', 'T5-Y0']
spgrid3=['Late M', 'L', 'T']


# In[16]:


sgrid,


# In[17]:


#for k in ['F140', 'F110', 'F160']:
#    tab['lim_{}'.format(k)]=tab.pnt.apply(lambda x: x.mag_limits[k])
#    tab['detected_{}'.format(k)]= tab[k+'W'] < tab['lim_{}'.format(k)]


# In[18]:


flags=np.array(flags)


# In[19]:


spgrid=np.arange(17, 42)


# In[20]:


fig, ax=plt.subplots(figsize=(12, 4), ncols=3)

ax[0].errorbar(tab.spt[flags], tab.F110W[flags], xerr=tab.spt_er[flags], yerr=tab.F110W_er[flags], fmt='o', c='k')
ax[0].errorbar(tab.spt[~flags], tab.F110W[~flags], xerr=tab.spt_er[~flags], yerr=tab.F110W_er[~flags],                 mfc='white', fmt='o')

ax[1].errorbar(tab.spt[flags], tab.F140W[flags], xerr=tab.spt_er[flags], yerr=tab.F140W_er[flags], fmt='o', c='k')
ax[1].errorbar(tab.spt[~flags], tab.F140W[~flags], xerr=tab.spt_er[~flags], yerr=tab.F140W_er[~flags],                 mfc='white', fmt='o')


ax[-1].errorbar(tab.spt[flags], tab.F160W[flags], xerr=tab.spt_er[flags], yerr=tab.F160W_er[flags], fmt='o', c='k')
ax[-1].errorbar(tab.spt[~flags], tab.F160W[~flags], xerr=tab.spt_er[~flags], yerr=tab.F160W_er[~flags],                 mfc='white', fmt='o')

for p in pnts:
    ax[0].plot(spgrid, p.mag_limits['F110']+(corr_pols['F110'+'W'][0])(spgrid), alpha=0.01, c='b')
    ax[1].plot(spgrid, p.mag_limits['F140']+(corr_pols['F140'+'W'][0])(spgrid), alpha=0.01, c='b')
    ax[-1].plot(spgrid, p.mag_limits['F160']+(corr_pols['F160'+'W'][0])(spgrid), alpha=0.01, c='b')

ax[0].set_xlabel('F110')
ax[1].set_xlabel('F140')
ax[-1].set_xlabel('F160')
ax[0].set_ylabel('SpT')
ax[1].set_ylabel('SpT')
ax[-1].set_ylabel('SpT')

for a in ax:
    a.minorticks_on()
plt.tight_layout()


# In[21]:


#wisps.POLYNOMIALS


# In[22]:


#.MAG_LIMITS


# In[23]:


subtab=(tab[tab.spt.between(30, 35)]).reset_index(drop=True)


# In[24]:


#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print( subtab[['F140W', 'F160W', 'lim_F140', 'lim_F160', 'detected_F140', 'detected_F160', 'grism_id',
#                  'spt']])


# In[ ]:





# In[25]:


#NUMBERS=pd.read_pickle(wisps.OUTPUT_FILES+'/numbers_simulated.pkl')
NUMBERS=get_all_numbers()


# In[26]:


NUMBERS.keys()


# In[27]:


#plt.hist(np.log10(NUMBERS['baraffe2003'][0][:,1]))


# In[28]:


volumes=[]
for pnt in pnts:
    vs=[]
    for h in wispsim.HS:
        vsx=[]
        for g in wispsim.SPGRID:
            vsx.append((pnt.volumes[h])[g])
        vs.append(vsx)
    volumes.append(vs)
volumes=np.array(volumes)

VOLUMES=(np.nansum(volumes, axis=0))*4.1*(u.arcmin**2).to(u.radian**2)


# In[29]:


MODEL_NAMES, MODEL_SHORT_NAMES


# In[30]:



def plot(NUMBERS, VOLUMES, filename='/oberved_numbers.pdf'):
    # In[ ]:
    nall=wisps.custom_histogram(cands.spt.apply(wisps.make_spt_number), sgrid, 1)
    
    y2=bin_by_spt_bin(wispsim.SPGRID,nobs, ltonly=False)-THICK
    yall=bin_by_spt_bin(wispsim.SPGRID,nall, ltonly=False)
    
    dy2=np.sqrt(y2)
    dyall=np.sqrt(yall)

    fig, ax=plt.subplots(figsize=(14, 8), ncols=3, nrows=2, sharey=True, sharex=False)
    
    for model, name, a in zip(MODEL_NAMES, MODEL_SHORT_NAMES, np.concatenate(ax)):
        
        for idx, h in enumerate(wispsim.HS):
            
            ns=None
            ns=((NUMBERS[model])[idx])[:,0]*VOLUMES[idx]
            nuncs=((NUMBERS[model])[idx])[:,1]*VOLUMES[idx]
            
            a.plot(spgrid2, bin_by_spt_bin(wispsim.SPGRID,ns, ltonly=False), 
                          color= cmap(cnorm(h)), 
                   linewidth=3, drawstyle="steps-mid")
            #a.fill_between(spgrid2, bin_by_spt_bin(wispsim.SPGRID,ns+nuncs, ltonly=False),  
             #              bin_by_spt_bin(wispsim.SPGRID,ns-nuncs, ltonly=False), alpha=0.5, 
             #              color= cmap(cnorm(h/100)),  step="mid")
        
        a.set_yscale('log')
        a.errorbar(spgrid2,y2, yerr=dy2,fmt='o', color='#111111')
        a.errorbar(spgrid2,yall, yerr=dyall,color='#B10DC9', mfc='white', fmt='o')
        a.set_xlabel('SpT',fontsize=18)
        a.set_ylabel('N',fontsize=18)
        a.minorticks_on()
        a.set_title('Model= {}'.format(name), fontsize=18)
            

    ax[1][-2].errorbar(spgrid2,y2, yerr=dy2,fmt='o', label='Mag Limited', color='#111111')
    ax[1][-2].errorbar(spgrid2,yall, yerr=dyall,color='#B10DC9', fmt='o', mfc='white', label='All Observations')
    
    #ax[-1][-2].legend(fontsize=14,  bbox_to_anchor=(1.05, 1), loc='upper left')
    #fig.delaxes(np.concatenate(ax)[-1])
    ax[1][-2].legend( fontsize=14, loc='upper right')
    
    cax = fig.add_axes([1.01, 0.25, .015, 0.5])
    mp=matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    cbar=plt.colorbar(mp, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(r'Scaleheight (H, pc)', fontsize=18)
    #cbar.ax.set_yticks([1, 3, 5, 10])
    #np.concatenate(ax)[-2].legend(loc='center left', bbox_to_anchor=(1, 1.5), fontsize=14)
    plt.tight_layout()
    plt.savefig(wisps.OUTPUT_FIGURES+filename, bbox_inches='tight')


# In[31]:


plot(NUMBERS, VOLUMES, filename='/obs_numbers_plus_binaries.pdf')


# In[32]:


#save into pickle file
#NUMBERS
counts_numbers={'volumes': VOLUMES, 'densities': NUMBERS, 'scaleheights': wispsim.HS, 'nobs': nobs}
import pickle
with open(wisps.OUTPUT_FILES+'/expected_numbers_wisps_plus_binaries.pkl', 'wb') as file:
    pickle.dump(counts_numbers,file)


# In[33]:


nall=wisps.custom_histogram(cands.spt.apply(wisps.make_spt_number), sgrid, 1)
y2=bin_by_spt_bin(wispsim.SPGRID,nobs, ltonly=False)-THICK


# In[34]:


def asymetric_errors(vals):
    if len(vals)<1:
        return [np.nan, np.nan]
    else:
        med= np.nanmedian(vals)
        up= np.nanpercentile(vals, 86)
        dn= np.nanpercentile(vals, 14)
        return np.array([med-dn, up-med])


# In[35]:


np.nanpercentile(wispsim.HS, 10)


# In[ ]:





# In[36]:


#just for L dwarfs and T dwarfs
y3=bin_by_spt_bin(wispsim.SPGRID,nall, ltonly=False)-THICK
y4=bin_by_spt_bin(wispsim.SPGRID,nobs, ltonly=True)#-THICK
y5= np.nansum(y4)
print ('all ----- {}'.format(y3))
print ('used ----- {}'.format(y2))
print ('MLT ----{}'.format(y4))
print ('All ----{}'.format(y5))


# In[37]:


#PRINT THE BEST FIT NUMBER 
#best_fit={}
numbers_fit={} #predictions for all
numbers_fit_lt={} #predictions for M, L, T
#numbers_fit_total={} #predictions for total number counts
for model in MODEL_NAMES:
        model_number_lt={}
        model_number={}
        for idx, h in enumerate(wispsim.HS):
            
            ns=None
            ns=((NUMBERS[model])[idx])[:,0]*VOLUMES[idx]
            nuncs=((NUMBERS[model])[idx])[:,1]*VOLUMES[idx]
            
            binned=np.array(bin_by_spt_bin(wispsim.SPGRID,ns, ltonly=False))
            binned_lt= np.array(bin_by_spt_bin(wispsim.SPGRID,ns, ltonly=True))
            #binned_unc=np.array(bin_by_spt_bin(wispsim.SPGRID,nuncs, ltonly=False))
            #add L and 
            #compute chi-squared
            #print (ns)
            #chisq= abs((y2-binned)**2/(y2))
            #model_fit.update({h: chisq})
            #binned_total=np.append(binned, binned_lt)
            #binned_total=np.append(binned, binned_lt)
            model_number.update({h: binned})
            model_number_lt.update({h: binned_lt})
        # best_fit.update({model: model_fit})
        numbers_fit.update({model: model_number})
        numbers_fit_lt.update({model:  model_number_lt})


# In[38]:


#chisq_dicts=pd.DataFrame.from_records(best_fit)
pred_number_dicts=pd.DataFrame.from_records(numbers_fit)
pred_number_lt_dicts=pd.DataFrame.from_records(numbers_fit_lt)


# In[39]:


from scipy.interpolate import interp1d


# In[40]:


import scipy.stats as stats


# In[41]:




def get_poisson_predictions(spt_grid, obstns, predns):
    res={}
    for c in  predns.columns:
        min_vals={}
        dist={}
        #for idx,s  in enumerate(np.append(spgrid2, ['L dwarfs', 'T dwarfs'])):
        for idx,s  in enumerate(spt_grid):
            #compare between subtypes
            #predicted
            predvals=(np.vstack(predns[c].values))[:,idx]
            #observed
            nreal=  obstns[idx]
            #make an interpolation function
            interpf = interp1d(predvals, wispsim.HS)
            #using a 2nd degree polynomial 
            #interpf = np.poly1d(np.polyfit(predvals, wispsim.HS, 3))

            #draw a bunch of random values based on a poisson distribution
            #npoisson=np.random.poisson(nreal, 100000).astype(float)
            #print (nreal)
            npoisson=stats.gamma.rvs(nreal, size =int(1e5))
            #stay within the range of possible values to avoid interpolation error
            #i.e take this as a prior
            #dflag=npoisson>=vals.min()
            #uflag= npoisson <=vals.max()
            #npoisson[dflag]= vals.min()
            #npoisson[uflag]= vals.max()
            #allow extraploayion
            npoisson=npoisson[np.logical_and(npoisson>=predvals.min(), npoisson <=predvals.max())]
            #predict scale heights
            predhs=interpf(npoisson)
            #use a weighted mean and std 
            #mean, unc= (np.nanmean(predhs), np.nanstd(predhs))

            #print (' scale height for model {} and spt {} is {} +/- {} '.format(c, s, np.round(mean), np.round(unc, 4)))
            dist.update({s:predhs})
        #min_chi_ssqrs.update({c:min_vals})
        res.update({c: dist})
    return  res


# In[42]:


def scaleheight_to_vertical_disp(hs):
    shape=435 #shape parameter
    sigma_68=1.
    return np.sqrt((np.array(hs))/shape)*20

def compute_age_with_z(sigmas, z):
    ag_bov= popsims.avr_yu(sigmas[abs(z) >270], verbose=True, nsample=2, height='above')[0]
    ag_bel=popsims.avr_yu(sigmas[abs(z) <=270], verbose=True, nsample=2, height='below')[0]
    return np.concatenate([ag_bov, ag_bel]).flatten()

def asssymetric_med_std(x):
    return np.round(np.nanmedian(x), 2), np.round(asymetric_errors(x), 2)


# In[43]:


def avr_aumer(sigma,  direction='vertical', verbose=False):
    #return the age from an age-velocity dispersion 
    verboseprint = print if verbose else lambda *a, **k: None
    result=None
    beta_dict={'radial': [0.307, 0.001, 41.899],
                'total': [ 0.385, 0.261, 57.15747],
                'azimuthal':[0.430, 0.715, 28.823],
                'vertical':[0.445, 0.001, 23.831],
                }

    verboseprint("Assuming Aumer & Binney 2009 Metal-Rich Fits and {} velocity ".format(direction))

    beta, tau1, sigma10=beta_dict[direction]
       
    result=((sigma/sigma10)**(1/beta))*(10+tau1)-tau1

    return result


# In[44]:


scale_height_dist=get_poisson_predictions(spgrid2, y2,pred_number_dicts)
scale_height_dist_df=pd.DataFrame(scale_height_dist)


# In[45]:


scale_height_dist_df['saumon2008']['M7-L0']=[]


# In[46]:


#scale_height_dist=get_poisson_predictions(spgrid2, y2,pred_number_dicts)
#scale_height_dist_df=pd.DataFrame(scale_height_dist)
vel_df=scale_height_dist_df.applymap(scaleheight_to_vertical_disp)
age_distdf_yu=vel_df.applymap(lambda x: popsims.avr_yu(x, verbose=False, nsample=2, height='median')[0])
age_distdf_just=vel_df.applymap(lambda x: popsims.avr_just(x, verbose=False))
age_distdf_sand=vel_df.applymap(lambda x: popsims.avr_sanders(x, verbose=False))
age_distdf_aumer=vel_df.applymap(lambda x: avr_aumer(x, verbose=False))



scalh_tables=scale_height_dist_df.applymap( asssymetric_med_std)
vel_tables=vel_df.applymap(asssymetric_med_std)
age_tables_yu=age_distdf_yu.applymap( asssymetric_med_std)
age_tables_just=age_distdf_just.applymap( asssymetric_med_std)
age_tables_sand=age_distdf_sand.applymap( asssymetric_med_std)
age_tables_aumer=age_distdf_aumer.applymap( asssymetric_med_std)


# In[47]:


import matplotlib as mpl
mpl.rcParams['figure.titlesize'] = 'large'


# In[48]:


pred_number_dicts.keys()


# In[49]:


idx=0
fig, ax=plt.subplots(figsize=(8, 6))
for m, ac in zip(['burrows1997', 'burrows2001', 'baraffe2003',], ['B97', 'B01', 'B03']):
    predvals=(np.vstack(np.vstack(pred_number_dicts[m].values)))[:,idx]
    #observed

    #make an interpolation function
    #interpf = np.poly1d(np.polyfit(predvals, wispsim.HS, 3))
    interpf =interp1d(predvals, wispsim.HS)

    rvs=stats.gamma.rvs(y2[idx], size =int(1e5))
    #rvs=rvs[np.logical_and(rvs>=predvals.min(), rvs <=predvals.max())]

    ax.plot(  wispsim.HS, predvals, marker='^', label=' {} Predictions'.format(ac))
    #ax[0].plot(  rvs, interpf(rvs), '.')
    
ax.axhline(y2[idx], color='r', label='Observations')
ax.axhspan(y2[idx]-np.sqrt(y2[idx]), y2[idx]+np.sqrt(y2[idx]), alpha=0.3, color='red')

ax.set(xlabel='Scaleheight (pc)', ylabel='Number Counts')


ax.minorticks_on()
ax.legend()
#fig.delaxes(ax0[1])
plt.tight_layout()
plt.savefig(wisps.OUTPUT_FIGURES+'/model_interpolation_ncounts.pdf', bbox_inches='tight')


# In[50]:


def reformat(val):
    return str(val[0])+'$ _{-'+str(val[1][0])+'} ^{+'+str(val[1][1])+'}$'


# In[51]:


scalh_tables[MODEL_NAMES].applymap(reformat)


# In[52]:


vel_tables[MODEL_NAMES].applymap(reformat)


# In[53]:


age_tables_just[MODEL_NAMES].applymap(reformat)


# In[54]:


#upper and lo limits on ages 
#up_lims_table=pd.DataFrame(columns= age_tables.columns,
#                           index=age_tables.index).fillna(0)
#up_lims_table.saumon2008['T0-T5']=1
#up_lims_table.saumon2008['T0-T5']=1

#lo limts
#lo_lims_table=pd.DataFrame(columns= age_tables.columns,
#                           index=age_tables.index).fillna(0)
#lo_lims_table.baraffe2003['T5-Y0']=1
#lo_lims_table.baraffe2003['L5-T0']=1
#lo_lims_table.phillips2020['T5-Y0']=1
#lo_lims_table.saumon2008['T5-Y0']=1
#lo_lims_table.marley2019['T5-Y0']=1


# In[ ]:


def get_simpler_class(x):
    if x.startswith('M'):
        return 'Late M'
    if x.startswith('L'):
        return 'L'
    if x.startswith('T'):
        return 'T'


# In[ ]:


#plot age with scale heights
age_dictionaries={}
for model in MODEL_NAMES:
    dfs=[]
    for hidx in wispsim.HS:
        #hidx is a scale height, model is evolutionary model
        df0=pd.read_hdf(wisps.OUTPUT_FILES+'/final_simulated_sample_cut_binaries.h5',                          key=str(model)+str(hidx)+str('spt_abs_mag'))
        cutdf=(df0[~df0.is_cut]).rename(columns={'temperature': 'teff',                                               'slprob': 'sl'})
        
        dfs.append(cutdf)
        
    df=pd.concat(dfs)
    print (len(df))
    cutdf_lblded=wisps.Annotator.group_by_spt(df, spt_label='spt', assign_number=False).rename(columns={'spt_range': 'subtype'})
    cutdf_lblded['spectclass']=  cutdf_lblded.subtype.apply(get_simpler_class)
    final_df=cutdf_lblded[~((cutdf_lblded.spectclass=='') | (cutdf_lblded.subtype=='')|   (cutdf_lblded.subtype=='trash'))]
    age_dictionaries[model]=final_df


# In[ ]:


#a.errorbar(  agfn.age, agfn.subtype, xerr=np.vstack(agfn.unc).T,  fmt='o',xlolims=lolims, ms=20, lw=7, 
#               capsize=7, 
#               mfc='#0074D9', mec='#111111', ecolor='#111111', xuplims=uplims)
#age_dictionaries[model].columns


# In[ ]:


#plot_one(NUMBERS, VOLUMES, filename='/oberved_numbers_one_panel.pdf')


# In[ ]:





# In[ ]:


def plot_one_age(a):
    model='baraffe2003'
    ds=[]
    dfn=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
    for k in age_tables_just[model].keys():
        if len(scale_height_dist[model][k]) <1:
            pass
        else:
            #empirical
            #ds0.append(age_tables_just[model][k])
            #from simulations
            ds.append(dfn.age[dfn.subtype==k].values)
    #some reformatting
    positions=[0, 1, 2, 3, 4]
    lolims=[0, 0, 0, 0, 0]
    if len(ds) ==4: positions=[1, 2, 3, 4]
    v1 = a.violinplot(ds,points=100, positions=positions,
               showmeans=True, showextrema=False, showmedians=False, vert =False)
    #for b in v1['bodies']: 
    #      b.set_color('#0074D9')
        #get the center
    xerr0=np.vstack((age_tables_just[model].apply(lambda x: x[1]).values)).T
    xerr1=np.vstack((age_tables_sand[model].apply(lambda x: x[1]).values)).T
    xerr2=np.vstack((age_tables_aumer[model].apply(lambda x: x[1]).values)).T
    
    #set size of arrows
    xerr0.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    xerr1.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    xerr2.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    
    #a.errorbar(age_tables_just[model].apply(lambda x: x[0]).values, [0, 1, 2, 3, 4],\
    #           xerr=xerr0,  fmt='o', label='J10', \
    #          ms=10, lw=5,  capsize=5,
    #           xlolims=lolims)
    
    #a.errorbar(age_tables_yu[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.2,\
    #           xerr=np.vstack((age_tables_yu[model].apply(lambda x: x[1]).values)).T,  fmt='o', label='Y18',
    #           ms=10, lw=5,  mfc='#B10DC9', mec='#B10DC9', ecolor='#B10DC9', capsize=5)
    
    #a.errorbar(age_tables_sand[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.0,\
    #           xerr=xerr1,  fmt='o', label='SB15',
    #           ms=10, lw=5,  mfc='#B10DC9', mec='#B10DC9', ecolor='#B10DC9', capsize=5,
    #            xlolims=lolims)
    
    #a.errorbar(age_tables_aumer[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.01,\
    #           xerr=xerr2,  fmt='o', label='AB09',
    #           ms=10, lw=5, capsize=5,
    #            xlolims=lolims)
    
    a.set_yticks([0, 1, 2, 3, 4])
    a.set_yticklabels(spgrid2)
    a.set_xlabel('Age (Gyr)', fontsize=18)
    a.set_ylabel('Subtype', fontsize=18)
    a.minorticks_on()
    a.set_xlim([-1, 10])
    a.set_title('Model= B03', fontsize=18)
    #a.legend(fontsize=12, loc='lower left')
   
    plt.tight_layout()
    


# In[ ]:


##plt.style.use('dark_background')
#fig, ax=plt.subplots(figsize=(6, 8))
#plot_one_age(ax)
#plt.savefig(wisps.OUTPUT_FIGURES+'/age_comparison_simsonly.pdf', bbox_inches='tight')


# In[ ]:


import scipy.stats as stats


# In[ ]:


fig, ax=plt.subplots(figsize=(12, 9), ncols=3, nrows=2, sharex=False, sharey=True)
for model, name, a in zip( MODEL_NAMES, MODEL_SHORT_NAMES, np.concatenate(ax)):

    ds=[]
    qs=[]
    
    dfn=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
    for k in age_tables_just[model].keys():
        if len(scale_height_dist[model][k]) <1:
            pass
            qs.append([np.nan, np.nan])
        else:
            #empirical
            #ds0.append(age_tables_just[model][k])
            #from simulations
            ds.append(dfn.age[dfn.subtype==k].values)
            qs.append(np.percentile(dfn.age[dfn.subtype==k].values, [16, 84]))
    #some reformatting
    qs=np.vstack(qs)
    a.hlines([0, 1, 2, 3, 4], qs[:,0], qs[:,1], color='#0074D9', linestyle='-', lw=3)
    positions=[0, 1, 2, 3, 4]
    lolims=[0, 0, 0, 0, 0]
    if len(ds) ==4: positions=[1, 2, 3, 4]
    v1 = a.violinplot(ds,points=100, positions=positions,
               showmeans=False, showextrema=False, showmedians=True, vert =False)
    for b in v1['bodies']: 
          b.set_color('#0074D9')
        #get the center
    xerr0=np.vstack((age_tables_just[model].apply(lambda x: x[1]).values)).T
    xerr1=np.vstack((age_tables_sand[model].apply(lambda x: x[1]).values)).T
    xerr2=np.vstack((age_tables_aumer[model].apply(lambda x: x[1]).values)).T
    
    #set size of arrows
    xerr0.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    xerr1.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    xerr2.T[np.array(lolims).astype(bool)]=[0.5, 0.5]
    
    a.errorbar(age_tables_just[model].apply(lambda x: x[0]).values, np.array([0, 1, 2, 3, 4])-0.2,               xerr=xerr0,  fmt='o', label='J10',               ms=10, lw=5,  mfc='#FF4136', mec='#FF4136', ecolor='#FF4136', capsize=5,
               xlolims=lolims)
    
    #a.errorbar(age_tables_yu[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.2,\
    #           xerr=np.vstack((age_tables_yu[model].apply(lambda x: x[1]).values)).T,  fmt='o', label='Y18',
    #           ms=10, lw=5,  mfc='#B10DC9', mec='#B10DC9', ecolor='#B10DC9', capsize=5)
    
    #a.errorbar(age_tables_sand[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.0,\
    #           xerr=xerr1,  fmt='o', label='SB15',
    #           ms=10, lw=5,  mfc='#B10DC9', mec='#B10DC9', ecolor='#B10DC9', capsize=5,
    #            xlolims=lolims)
    
    a.errorbar(age_tables_aumer[model].apply(lambda x: x[0]).values,  np.array([0, 1, 2, 3, 4])+0.2,               xerr=xerr2,  fmt='o', label='AB09',
               ms=10, lw=5,  mfc='#111111', mec='#111111', ecolor='#111111', capsize=5,
                xlolims=lolims)
    
    a.set_yticks([0, 1, 2, 3, 4])
    a.set_yticklabels(spgrid2)
    a.set_xlabel('Age (Gyr)', fontsize=18)
    a.set_ylabel('Subtype', fontsize=18)
    a.minorticks_on()
    a.set_ylim([-1, 5])
    a.set_title('Model= {}'.format(name), fontsize=18)
    
ax[0][-2].legend(fontsize=12, loc='lower left')

plt.tight_layout()
plt.savefig(wisps.OUTPUT_FIGURES+'/age_comparison_plus_binaries.pdf', bbox_inches='tight',              facecolor='white', transparent=True)


# In[ ]:


scale_height_dist['saumon2008']['M7-L0']=[]


# In[ ]:


np.shape(ds)


# In[ ]:


#fig, ax=plt.subplots()
np.percentile((scale_height_dist[model][k]).flatten(), [25,75], axis=0)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.percentile')


# In[ ]:


fig, ax=plt.subplots(figsize=(12, 8), ncols=3, nrows=2, sharex=False, sharey=True)

for model, name, a in zip( MODEL_NAMES, MODEL_SHORT_NAMES, np.concatenate(ax)):
    ds=[]
    qs=[]
    for k in scale_height_dist[model].keys():
        if len(scale_height_dist[model][k]) <1:
            pass
            qs.append([np.nan, np.nan])
        else:
            ds.append(scale_height_dist[model][k])
            qs.append(np.percentile(scale_height_dist[model][k], [16, 84]))
            
    qs=np.vstack(qs)
    a.hlines([0, 1, 2, 3, 4], qs[:,0], qs[:,1], color='r', linestyle='-', lw=3)

    positions=[0, 1, 2, 3, 4]
    if len(ds) ==4: positions=[1, 2, 3, 4]
    v1 = a.violinplot(ds,points=3000, positions=positions,
               showmeans=False, showmedians=True,  \
                      showextrema=False, vert =False)
    for b in v1['bodies']: 
          b.set_color('#FF4136')
        #get the center
    a.set_yticks([0, 1, 2, 3, 4])
    a.set_yticklabels(spgrid2)
    a.set_xlabel('H (pc)', fontsize=18)
    a.set_ylabel('Subtype', fontsize=18)
    a.minorticks_on()
    a.set(xlim=[0, 500])
    a.set_title('Model= {}'.format(name), fontsize=18)


plt.tight_layout()
plt.savefig(wisps.OUTPUT_FIGURES+'/scaleheight_comparison_plus_binaries.pdf', bbox_inches='tight',              facecolor='white', transparent=True)


# In[ ]:


ls=np.concatenate([np.concatenate(scale_height_dist_df.loc['L0-L5'].values).flatten(),                  np.concatenate(scale_height_dist_df.loc['L5-T0'].values).flatten()])

ts=np.concatenate([np.concatenate(scale_height_dist_df.loc['T0-T5'].values).flatten(),                  np.concatenate(scale_height_dist_df.loc['T5-Y0'].values).flatten()])


# In[ ]:


print ('T ', asssymetric_med_std(ts))



print ('L ', asssymetric_med_std(ls))





print ('M ',asssymetric_med_std(np.concatenate(scale_height_dist_df.loc['M7-L0'].values).flatten()))





print ('T ',asssymetric_med_std(np.concatenate(scale_height_dist_df.loc['M7-L0'].values).flatten()))




ignore_models_dict={'L0-L5': ['phillips2020'], 'M7-L0':[], 'L5-T0':[],                    'T0-T5':[], 'T5-Y0':[],  'T5-Y0':[]}




def get_median(df, subtype, rund=1):
    ignore_models=ignore_models_dict[subtype]
    all_models=MODEL_NAMES
    mds=[x for x in all_models if x not in ignore_models]
    vs=np.concatenate(df[mds].loc[subtype].values).flatten()
    val, unc=asssymetric_med_std(vs)
    res=''
    if np.isnan(val):
            res += r'\nodata'
    else:
        if rund <1:
                st=str(int(np.round(val, rund)))+ '$_{-'+ str(int(np.round(unc[0], rund)))+'}'+                '^{+'+str(int(np.round(unc[1], rund)))+'} $&'
        else:
                st=str(np.round(val, rund))+ '$_{-'+ str(np.round(unc[0], rund))+'}'+                '^{+'+str(np.round(unc[1], rund))+'} $&'
        res += st
    
    return res
def get_formatted_string(df, subtype, rund=1):
    dn=df.loc[subtype]
    res=''
    for md in MODEL_NAMES:
        val= dn[md][0]
        unc=dn[md][1]
        if np.isnan(val):
            res += r'\nodata &'
        else:
            if rund <1:
                st=str(int(np.round(val, rund)))+ '$_{-'+ str(int(np.round(unc[0], rund)))+'}'+                '^{+'+str(int(np.round(unc[1], rund)))+'} $&'
            else:
                st=str(np.round(val, rund))+ '$_{-'+ str(np.round(unc[0], rund))+'}'+                '^{+'+str(np.round(unc[1], rund))+'} $&'
            res += st
    return res

def get_age_median_from_simulation(subtype, rund=1):
    res=''
    
    for model in MODEL_NAMES:
        dfn0=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
        dfn=dfn0[dfn0.subtype==subtype]
        if len(scale_height_dist[model][subtype]) <1:
            res += r'\nodata &'
        else:
            val, unc= asssymetric_med_std(dfn['age'].values)
            if rund <1:
                st=str(int(np.round(val, rund)))+ '$_{-'+ str(int(np.round(unc[0], rund)))+'}'+                '^{+'+str(int(np.round(unc[1], rund)))+'} $&'
            else:
                st=str(np.round(val, rund))+ '$_{-'+ str(np.round(unc[0], rund))+'}'+                '^{+'+str(np.round(unc[1], rund))+'} $&'
            res += st
    return res

def get_age_median_all_from_simulation(subtype, rund=1):
    ignore_models=ignore_models_dict[subtype]
    ds=[]
    res=''
    all_models=MODEL_NAMES
    mds=[x for x in all_models if x not in ignore_models]
    for model in mds:
        dfn0=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
        ds.append(dfn0[dfn0.subtype==subtype].age.values)
    val, unc=asssymetric_med_std(np.concatenate(ds))
    st=str(np.round(val, rund))+ '$_{-'+ str(np.round(unc[0], rund))+'}'+                '^{+'+str(np.round(unc[1], rund))+'} $&'
    res += st
    return res

def get_ks_stats(subtype, rund=1, compareto='aumer'):
    res=''
    for model in MODEL_NAMES:
        dfn=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
        x=dfn[dfn.subtype ==subtype].age.values
        if compareto=='aumer': y=age_distdf_aumer[model][subtype]
        if compareto=='just': y=age_distdf_just[model][subtype]
        if len(y)<2:
            res +=r'\nodata'
        else:
            val=stats.ks_2samp(x, y, mode='asymp', alternative='two-sided')[0]
            st= str(np.round(val, rund))+'&'
            res += st
    return res


# In[ ]:


def custom_overlap_probability(subtype, ranges=[0, 13], rund=3, compareto='aumer'):
    grid=np.linspace(ranges[-1], ranges[1], 100)
   
    res=''
    for model in MODEL_NAMES:
        dfn=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
        x=dfn[dfn.subtype ==subtype].age.values
        if compareto=='aumer':y=age_distdf_aumer[model][subtype]
        if compareto=='just': y=age_distdf_just[model][subtype]
        if len(y)<2:
            res +='\nodata'
        else:
            #Create overlapping probabilities
            kde_sim=stats.kde.gaussian_kde(x)
            kde_obs=stats.kde.gaussian_kde(y)

            num=np.trapz(kde_sim(grid)*kde_obs(grid), x=grid)
            den=np.trapz(kde_sim(grid), x=grid)*np.trapz(kde_obs(grid), x=grid)
            val=num/den
            st= str(np.round(val, rund))+'&'
            res += st
    
    return res
                                                
                            


# In[ ]:


grid=np.linspace(0, 13, 5000 )
x=np.random.normal(5, 1, 1000)
y=np.random.normal(5, 1, 1000)


# In[ ]:


kde_sim=stats.kde.gaussian_kde(x)
kde_obs=stats.kde.gaussian_kde(y)

num=np.trapz(kde_sim(grid)*kde_obs(grid), x=grid)
den=np.trapz(kde_sim(grid), x=grid)*np.trapz(kde_obs(grid), x=grid)


# In[ ]:


plt.plot(grid, kde_sim(grid))
plt.plot(grid, kde_obs(grid))


# In[ ]:


num/den


# In[ ]:


stats.ks_2samp(x, y, mode='asymp', alternative='two-sided')[0]


# In[ ]:


#print latex formatted 
for idx, subtype in enumerate(spgrid2):
    print (subtype + r'&$H$ (pc) &' + get_formatted_string(scalh_tables, subtype, rund=0)            + get_median(scale_height_dist_df, subtype, rund=0) +             str(int(np.round(y2)[idx]))+'&'+  str(np.round(THICK[idx], 1)) + r'\\ ')
    print (r' & $\sigma_w$ (km/s)  &' + get_formatted_string(vel_tables, subtype)            + get_median(vel_df, subtype) + '& '+ r'\\ ')
    print (r' & Age (Gyr) (J10) &' + get_formatted_string(age_tables_just, subtype)            + get_median(age_distdf_just, subtype)+ '& ' + r'\\ ')
    #print (r' & Age (Gyr) (SB15)&' + get_formatted_string(age_tables_sand, subtype)  \
    #       + get_median(age_distdf_sand, subtype) + r'\\ ')
    #print (r' & Age (Gyr) (Y18)&' + get_formatted_string(age_tables_yu, subtype)  \
    #       + get_median(age_distdf_yu, subtype) + r'\\ ')
    print (r' & Age (Gyr) (A09)&' + get_formatted_string(age_tables_aumer, subtype)             + get_median(age_distdf_aumer, subtype)+ '& '  + r'\\ ')
    
    print (r' & Age (Gyr) (Simulation)&' + get_age_median_from_simulation(subtype, rund=1)             + get_age_median_all_from_simulation(subtype, rund=1)+ '& ' + r'\\ ')
    
    print (r' & KS (A09-Simulation) & ' + get_ks_stats(subtype, rund=1, compareto='aumer')           +  '& ' + r'\\ ')
    
    print (r' & KS (J10-Simulation) & ' + get_ks_stats(subtype, rund=1, compareto='just')           +  '& ' + r'\\ ')


# In[ ]:


###################run same analysis for combined L and T d



ltotal_scaleheight={}
ttotal_scaleheight={}


# In[ ]:


#compute rvs for 
fig, ax=plt.subplots()
for m in MODEL_NAMES:
    predvals=np.vstack(np.vstack(pred_number_dicts[m].values))[:,1]+    np.vstack(np.vstack(pred_number_dicts[m].values))[:,2]
    interpf =interp1d(predvals, wispsim.HS)
    rvs=stats.gamma.rvs(17, size =int(1e5))
    npoisson=rvs[np.logical_and(rvs>=predvals.min(), rvs <=predvals.max())]
    ltotal_scaleheight.update({m: interpf(npoisson)})
    ax.plot(npoisson, interpf(npoisson), '^', alpha=0.1)
    
#compute rvs for 
for m in MODEL_NAMES:
    predvals=np.vstack(np.vstack(pred_number_dicts[m].values))[:,3]+    np.vstack(np.vstack(pred_number_dicts[m].values))[:,4]
    interpf =interp1d(predvals, wispsim.HS)
    rvs=stats.gamma.rvs(4, size =int(1e5))
    npoisson=rvs[np.logical_and(rvs>=predvals.min(), rvs <=predvals.max())]
    ttotal_scaleheight.update({m: interpf(npoisson)})


# In[ ]:


ltotal_scaleheight.keys()


# In[ ]:


ltotal_scaleheight['combined']=np.concatenate([ltotal_scaleheight[k]                                 for k in ltotal_scaleheight.keys()])
ttotal_scaleheight['combined']=np.concatenate([ttotal_scaleheight[k]                                 for k in ttotal_scaleheight.keys()])


# In[ ]:


ttotal_vdisp={}
ltotal_vdisp={}

ttotal_age_just={}
ttotal_age_aumer={}

ltotal_age_just={}
ltotal_age_aumer={}
for k in ttotal_scaleheight.keys():
    ltotal_vdisp.update({k:scaleheight_to_vertical_disp(ltotal_scaleheight[k])})
    ttotal_vdisp.update({k:scaleheight_to_vertical_disp(ttotal_scaleheight[k])})
    
    ttotal_age_just.update({k: popsims.avr_just(ttotal_vdisp[k], verbose=False)})
    ttotal_age_aumer.update({k: avr_aumer(ttotal_vdisp[k], verbose=False)})
    
    ltotal_age_just.update({k:  popsims.avr_just(ltotal_vdisp[k], verbose=False)})
    ltotal_age_aumer.update({k:avr_aumer(ltotal_vdisp[k], verbose=False)})
    
    


# In[ ]:


lt_df_h=pd.DataFrame(columns= ttotal_scaleheight.keys(), index=['L', 'T'])
lt_df_age_aumer=pd.DataFrame(columns= ttotal_scaleheight.keys(), index=['L', 'T'])
lt_df_age_just=pd.DataFrame(columns= ttotal_scaleheight.keys(), index=['L', 'T'])
lt_df_v=pd.DataFrame(columns= ttotal_scaleheight.keys(), index=['L', 'T'])





for k in ttotal_scaleheight.keys():
    lt_df_h.loc['L', k]=asssymetric_med_std(ltotal_scaleheight[k])
    lt_df_v.loc['L', k]=asssymetric_med_std(ltotal_vdisp[k])
    lt_df_age_aumer.loc['L', k]=asssymetric_med_std(ltotal_age_aumer[k])
    lt_df_age_just.loc['L', k]=asssymetric_med_std(ltotal_age_just[k])
    
    lt_df_h.loc['T', k]=asssymetric_med_std(ttotal_scaleheight[k])
    lt_df_v.loc['T', k]=asssymetric_med_std(ttotal_vdisp[k])
    lt_df_age_aumer.loc['T', k]=asssymetric_med_std(ttotal_age_aumer[k])
    lt_df_age_just.loc['T', k]=asssymetric_med_std(ttotal_age_just[k])





print (lt_df_h.applymap(reformat))




print (lt_df_v.applymap(reformat))





print (lt_df_age_aumer.applymap(reformat))





def get_combined_age_lt_simulation():
    comb_dt={'L':[], 'T':[]}
    combined_sim_ages_dict={'L':{}, 'T':{}}
    for model in tqdm(MODEL_NAMES):
        dfn0=age_dictionaries[model].replace('T5-T9', 'T5-Y0')
        ls=dfn0[dfn0.spt.between(20, 29)].age.values
        ts=dfn0[dfn0.spt.between(30, 39)].age.values
        combined_sim_ages_dict['L'].update({model:asssymetric_med_std(ls) })
        combined_sim_ages_dict['T'].update({model:asssymetric_med_std(ts) })
        comb_dt['L'].append(ls)
        comb_dt['T'].append(ts)
    
    df=pd.DataFrame.from_records(  combined_sim_ages_dict).T
    #print (asssymetric_med_std(np.concatenate(comb_dt['L'])))
    #df.loc['L', 'Combined']=asssymetric_med_std(np.concatenate(comb_dt['L']))
    #df.loc['T', 'Combined']=asssymetric_med_std(np.concatenate(comb_dt['T']))
    return df, [ asssymetric_med_std(np.concatenate(comb_dt['L'])),               asssymetric_med_std(np.concatenate(comb_dt['T']))]
        
    





comb_ages_lt,comb= get_combined_age_lt_simulation()





comb_ages_lt['combined']=comb





for k in ['L', 'T']:
    res=''
    for m, mshort in zip(np.append(MODEL_NAMES, 'combined'),                         np.append(MODEL_SHORT_NAMES, 'Combined')):
        val, unc=(lt_df_h.loc[k, m])
        val1, unc1=(lt_df_v.loc[k, m])
        val2, unc2=(lt_df_age_aumer.loc[k, m])
        val3, unc3=(lt_df_age_just.loc[k, m])
        val4, unc4=(comb_ages_lt.loc[k, m])
        st= str(int(val))+ '$_{-'+str(int(unc[0]))+'}'        + '^{+'+str(int(unc[-1]))+'}'+r' $&'
        
        st1= str(round(val1, 1))+ '$_{-'+str(round(unc1[0], 1))+'}'        + '^{+'+str(round(unc1[-1], 1))+'}'+r' $&'
        
        st2= str(round(val2, 1))+ '$_{-'+str(round(unc2[0], 1))+'}'        + '^{+'+str(round(unc2[-1], 1))+'}'+r' $& '
        
        st3= str(round(val3, 1))+ '$_{-'+str(round(unc3[0], 1))+'}'        + '^{+'+str(round(unc3[-1], 1))+'}$'
        
        st4= str(round(val4, 1))+ '$_{-'+str(round(unc4[0], 1))+'}'        + '^{+'+str(round(unc4[-1], 1))+'}$'

        res += '&'+ mshort + '&'+ st+ st1+st2+st3+'&'+ st4+ r'\\'
    print (res)
    







