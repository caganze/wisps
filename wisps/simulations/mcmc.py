from .initialize import OUTPUT_FILES, LUMINOSITY_FUCTION
from .core import *
import pandas as pd
import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import corner
import pickle
from scipy import interpolate
from tqdm import tqdm
#
measured_luminosity=pd.read_pickle(OUTPUT_FILES+'/measured_lf.pickle')

def lnlike(theta, model_input):
	"""
	likelihood function
	"""
	sh, sl=theta

	spts=model_input['spts']
	coords=model_input['coords']
	dmins=model_input['dmins']
	dmaxs=model_input['dmaxs']
	lf=model_input['lnew']
	area=model_input['area']
	vols=[]
	for spt, dmin, dmax in  tqdm(zip(spts, dmins, dmaxs)):
		vols.append([volume(c, sh, sl, di*u.pc, da*u.pc, area).value for c, di, da in zip(coords, dmin, dmax)])

	lfsim=lf*np.nansum(np.array(vols), axis=1)
	return -np.nansum((lfsim-measured_luminosity.number)**2)

def lnprior(theta):
    sh, sl=theta
    if .0 < sl< 100000 and 0.0 < sh < 100000.0 :
        return 0.0
    return -np.inf

def lnprob(theta, model_input):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model_input)

def run_mcmc(sh0=3000.0, sl0=2000.0, ndim=2, nwalkers=100):
   	"""
   	Run the MCMC based on a prior yos
   	"""
   	#read
   	with open(OUTPUT_FILES+'/model_input.pickle', 'rb') as f:
   		model_input = pickle.load(f)

 
   	#initial positions for each walker
   	pos = [[sh0, sl0]+100*np.random.randn(ndim) for i in range(nwalkers)]
   	#Then, we can set up the sampler
   	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[model_input])
   	#and run the MCMC for 500 steps starting from the tiny ball defined above:
   	sampler.run_mcmc(pos, 500)

   	fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])

   	fig.savefig(OUTPUT_FILES+"triangle.pdf")
   	return 
