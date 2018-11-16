
# coding: utf-8

# # code experimenting with surface density measurements from WISP

# In[1]:


import splat
import splat.evolve as spev
import splat.empirical as spem
import splat.simulate as spsim
import splat.plot as splot
import splat.photometry as sphot
import matplotlib.pyplot as plt
import numpy
import pandas
import time
from astropy.coordinates import SkyCoord
splat.initializeStandards()
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import wisps
import numpy as np
candidates=pd.read_pickle(wisps.OUTPUT_FILES+'/candidates.pkl')


# In[3]:


mags=pd.DataFrame([x.mags for x in candidates]).applymap(lambda x: np.array(x)[0])
min_ff10w_mag=np.nanmax(mags.F110W)
min_ff60w_mag=np.nanmax(mags.F160W)
min_ff40w_mag=np.nanmax(mags.F140W)

print (min_ff10w_mag, min_ff60w_mag, min_ff60w_mag)


# In[4]:


# initialize
dspt = pandas.DataFrame()
dspt['sptn'] = numpy.arange(20,39)
dspt['spt'] = [splat.typeToNum(x) for x in dspt['sptn']]


# In[5]:


# LF from Burgasser (2007) for dn/dM ~ M^-0.5
# can replace this with simulated LF (next box)
#lf = numpy.array([.637,.861,1.,.834,.819,.869,.785,.644,.462,.308,.22,.241,.406,.788,1.42,2.38,3.45,4.82])*1.e-3
#dspt['lf'] = lf


# In[6]:


# ALTERNATE: generate LF from simulation, normalized in 0.09-0.1 Msun range - this allows for more flexibility
norm_range = [0.09,0.1]
norm_density = 0.0037
nsim = 1e4
# simulation
masses = spsim.simulateMasses(nsim,range=[0.02,0.15],distribution='power-law',alpha=0.5)
ages = spsim.simulateAges(nsim,range=[0.1,10.],distribution='uniform')
teffs = spev.modelParameters(mass=masses,age=ages,set='baraffe03')['temperature'].value
spts = numpy.array([spem.typeToTeff(float(x),set='filippazzo',reverse=True)[0] for x in teffs])
norm = norm_density/len(masses[numpy.where(numpy.logical_and(masses>=norm_range[0],masses<norm_range[1]))])


# In[7]:


# generate binned luminosity function
lfsim = []
spts = spts[numpy.isfinite(spts) == True]
for x in dspt['sptn']: lfsim.append(len(spts[numpy.where(numpy.logical_and(spts>=x,spts<x+1.))]))
lfsim = numpy.array(lfsim)*norm
dspt['lf'] = lfsim

plt.figure(figsize=[8,4])
plt.step(dspt['spt'],lfsim,label='SPLAT Simulation')
plt.step(dspt['spt'],lf,color='r',label='Burgasser (2007)')
plt.xlabel('Spectral Type')
plt.ylabel(r'Number Density (pc$^{-3}$ SpT$^{-1}$)')
plt.legend()


# In[8]:


import wisps
wisp = pandas.read_csv(wisps.OUTPUT_FILES+'/observation_log_with_mag.csv')


# In[9]:


# WISP parameters
dp = pandas.DataFrame()

dp['pointing'] = wisp['POINTING']
dp['ra'] = wisp['RA (deg)']
dp['dec'] = wisp['DEC(deg)']
dp['coordinates'] = SkyCoord(ra=wisp['RA (deg)'],dec=wisp['DEC(deg)'],unit='degree')
dp['area'] = wisp['AREA (arcmin^2)']*((1./60.)*(numpy.pi/180.))**2
# choose some random magnitude limits (replace with actual limits)
# these don't seem to be right
dp['f110_lim_faint'] = 24.0
dp['f110_lim_faint'][numpy.isfinite(dp['f110_lim_faint']) == False] = 99.
dp['f140_lim_faint'] = 24.0
dp['f140_lim_faint'][numpy.isfinite(dp['f140_lim_faint']) == False] = 99.
dp['f160_lim_faint'] =24.0
dp['f160_lim_faint'][numpy.isfinite(dp['f160_lim_faint']) == False] = 99.
# remove any fields that have no limiting magnitudes
dp['maglimmax'] = dp.loc[:, ['f110_lim_faint', 'f140_lim_faint', 'f160_lim_faint']].min(axis=1)
l = len(dp)
dp = dp[dp['maglimmax'] < 99.]
dp.reset_index(inplace=True)
ll = len(dp)
if ll < l: print('Dropped {} fields without limiting magnitudes'.format(l-ll))
# these don't seem to be right
dp['f110_lim_bright'] = wisp['MAX_F160']
dp['f140_lim_bright'] = wisp['MAX_F160']
dp['f160_lim_bright'] = wisp['MAX_F160']
#dp['f110_lim_faint'] = numpy.random.uniform(24.5,25.5,len(dp))
#dp['f140_lim_faint'] = numpy.random.uniform(24.5,25.5,len(dp))
#dp['f160_lim_faint'] = numpy.random.uniform(24.5,25.5,len(dp))
#dp['f110_lim_bright'] = numpy.random.uniform(13.5,14.5,len(dp))
#dp['f140_lim_bright'] = numpy.random.uniform(13.5,14.5,len(dp))
#dp['f160_lim_bright'] = numpy.random.uniform(13.5,14.5,len(dp))


# In[10]:


wisp['MAX_F110'].plot(kind='hist', alpha=0.9, color='#001f3f', label='bright F110')
wisp['MAX_F160'].plot(kind='hist', alpha=0.9, color='#7FDBFF', label='bright F160')
wisp['MAX_F140'].plot(kind='hist', alpha=0.9, color='#2ECC40', label='bright F140')
plt.legend()


# In[11]:


# determine color terms and distance limits as a function of spectral type
dspt['j-f110'] = [sphot.filterMag(splat.STDS_DWARF_SPEX[x],'2MASS J')[0]-sphot.filterMag(splat.STDS_DWARF_SPEX[x],'F110W')[0] for x in dspt['spt']]
dspt['j-f140'] = [sphot.filterMag(splat.STDS_DWARF_SPEX[x],'2MASS J')[0]-sphot.filterMag(splat.STDS_DWARF_SPEX[x],'F140W')[0] for x in dspt['spt']]
dspt['h-f140'] = [sphot.filterMag(splat.STDS_DWARF_SPEX[x],'2MASS H')[0]-sphot.filterMag(splat.STDS_DWARF_SPEX[x],'F140W')[0] for x in dspt['spt']]
dspt['h-f160'] = [sphot.filterMag(splat.STDS_DWARF_SPEX[x],'2MASS H')[0]-sphot.filterMag(splat.STDS_DWARF_SPEX[x],'F160W')[0] for x in dspt['spt']]
# absolute magnitudes
dspt['absj'] = spem.typeToMag(dspt['spt'],'2MASS J',set='dupuy')[0]
dspt['absh'] = spem.typeToMag(dspt['spt'],'2MASS J',set='dupuy')[0]
dspt['absf110'] = dspt['absj']-dspt['j-f110']
dspt['absf140_1'] = dspt['absj']-dspt['j-f140']
dspt['absf140_2'] = dspt['absh']-dspt['h-f140']
dspt['absf160'] = dspt['absh']-dspt['h-f160']


# In[12]:


# determine distance limits as a function of spectral type
dspt['avg_dmax'] = numpy.zeros(len(dspt))
dspt['avg_dmin'] = numpy.zeros(len(dspt))
for i,x in enumerate(dspt['spt']):
# maximum distance - take the minimum from each filter/color
    dtmp = pandas.DataFrame()
    dtmp['df110'] = 10.*10.**(0.2*(dp['f110_lim_faint']-dspt['absf110'].iloc[i]))
    dtmp['df140_1'] = 10.*10.**(0.2*(dp['f140_lim_faint']-dspt['absf140_1'].iloc[i]))
    dtmp['df140_2'] = 10.*10.**(0.2*(dp['f140_lim_faint']-dspt['absf140_2'].iloc[i]))
    dtmp['df160'] = 10.*10.**(0.2*(dp['f160_lim_faint']-dspt['absf160'].iloc[i]))
    dp['{}_dmax'.format(x)] = dtmp.loc[:, ['df110', 'df140_1', 'df140_2','df160']].min(axis=1)
    del dtmp
# minimum distance - take the maximum from each filter/color
    dtmp = pandas.DataFrame()
    dtmp['df110'] = 10.*10.**(0.2*(dp['f110_lim_bright']-dspt['absf110'].iloc[i]))
    dtmp['df140_1'] = 10.*10.**(0.2*(dp['f140_lim_bright']-dspt['absf140_1'].iloc[i]))
    dtmp['df140_2'] = 10.*10.**(0.2*(dp['f140_lim_bright']-dspt['absf140_2'].iloc[i]))
    dtmp['df160'] = 10.*10.**(0.2*(dp['f160_lim_bright']-dspt['absf160'].iloc[i]))
    dp['{}_dmin'.format(x)] = dtmp.loc[:, ['df110', 'df140_1', 'df140_2','df160']].max(axis=1)
    del dtmp
# average values
    dspt['avg_dmax'].iloc[i] = numpy.nanmean(numpy.array(dp['{}_dmax'.format(x)]))
    dspt['avg_dmin'].iloc[i] = numpy.nanmean(numpy.array(dp['{}_dmin'.format(x)]))


# In[13]:


# check these distances
import warnings
warnings.filterwarnings('ignore')
for i in range(len(dp)):
    dx = [dp['{}_dmax'.format(x)].iloc[i] for x in dspt['spt']]
    plt.semilogy(dspt['spt'],dx,'b-',alpha=0.05)
    dn = [dp['{}_dmin'.format(x)].iloc[i] for x in dspt['spt']]
    plt.semilogy(dspt['spt'],dn,'r-',alpha=0.05)
plt.semilogy(dspt['spt'],dspt['avg_dmax'],'k-')
plt.semilogy(dspt['spt'],dspt['avg_dmin'],'k-')


# In[14]:


# volume corrections and volume sampled - takes several minutes
# timing test
t = time.time()
for i,c in enumerate(dp['coordinates'][:10]):
    x = 'L0.0'
    tmp = spsim.volumeCorrection(c,dp['{}_dmax'.format(x)].iloc[i],dmin=dp['{}_dmin'.format(x)].iloc[i])
print('Total estimated time = {} minutes'.format((time.time()-t)/10./60.*len(dp)*len(dspt)*2.))
for j,x in enumerate(dspt['spt']):
    t = time.time()
    dp['{}_vc'.format(x)] = numpy.zeros(len(dp))
    for i,c in enumerate(dp['coordinates']):
        dp['{}_vc'.format(x)].iloc[i] = spsim.volumeCorrection(c,dp['{}_dmax'.format(x)].iloc[i],dmin=dp['{}_dmin'.format(x)].iloc[i])
    dp['{}_volume'.format(x)] = dp['{}_vc'.format(x)]*(1./3.)*(dp['{}_dmax'.format(x)]**3-dp['{}_dmin'.format(x)]**3)*dp['area']
    print('Finished {} in {} seconds'.format(x,time.time()-t))


# In[15]:


#dspt['spt']=dspt['spt'].apply(lambda x: x.split('.')[0])


# In[16]:


# check these volumes
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 17}

import matplotlib
matplotlib.rc('font', **font)
plt.figure(figsize=(12, 12))
dspt['volume'] = numpy.zeros(len(dspt))
for i,x in enumerate(dspt['spt']):
    dspt['volume'].iloc[i] = numpy.nansum(numpy.array(dp['{}_volume'.format(x)]))
for i in range(len(dp)):
    v = [dp['{}_volume'.format(x)].iloc[i] for x in dspt['spt']]
    plt.semilogy(dspt['spt'],v,'b-',alpha=0.05)
plt.semilogy(dspt['spt'],dspt['volume'],'k-')
plt.xlabel('SpT')
plt.ylabel('Volumes sampled')


# In[17]:


import pandas as pd
candidates=pd.read_pickle(wisps.OUTPUT_FILES+'/candidates.pkl')


# In[18]:


spts=[splat.typeToNum(s.spectral_type) for s in candidates  ]

nspts=[x for x in spts if x>=20.0]
nspts2=[splat.typeToNum(x) for x in np.arange(21, 39)]


# In[19]:


import numpy as np
hist=np.histogram(nspts, bins=39-21)


# In[24]:


plt.hist(spts)


# In[21]:


# compute surface densities/predicted numbers
#matplotlib default font
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
dspt['n_predict'] = dspt['volume']*dspt['lf']

import matplotlib
matplotlib.rc('font', **font)
fig, ax=plt.subplots(figsize=[14, 8])
ax.step(dspt['spt'],dspt['n_predict'], label='predicted', color='#0074D9')
ax.step(dspt['spt'],dspt['n_predict']*0.9, label='predicted*completenes', color='#2ECC40')
ax.step(  nspts2, hist[0], label='measured', color='#111111')

plt.xlabel('Spectral Type')
plt.ylabel('Number/SpT bin')
#plt.tight_layout()
plt.legend()
print(numpy.sum(numpy.array(dspt['n_predict'])))


# In[22]:


# save spreadsheets
dp.to_excel('survey_data.xlsx',index=False)
dspt.to_excel('spt_data.xlsx',index=False)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
s.plot(filename=wisps.OUTPUT_FIGURES+'/spectra/'+s.name+'.jpg')


# In[ ]:


import splat
import splat.evolve as spev
spev.modelParameters('baraffe03',teff=2419.93, age=6.5)


# In[ ]:


import splat


# In[ ]:


splat.checkAccess()

