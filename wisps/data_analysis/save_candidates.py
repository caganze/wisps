from .initialize import *
import pandas as pd
import numpy as np
import glob
from .photometry import Source
from .spectrum_tools import Spectrum



import numba

cands=pd.read_pickle(OUTPUT_FILES+'/selected_by_indices.pkl')
import os
SPECTRA_PATH=os.path.dirname(WISP_PATH.split('wisps')[0]+('wisps')+'//spectra//')


# a lot of routines to make my visual inspection of candidates a feedback loop
def plot_name(name):
	fname=SPECTRA_PATH+'/indices/'+name.replace('-', '_')+'.jpeg'
	#plot(name, fname)
	if os.path.isfile(fname) : pass
	else: plot(name, fname)
	
def plot(n, fname):
	try:
		print (n.strip().replace('g141', 'G141'))
		s=Source(name=n.strip().replace('g141', 'G141'))
		s.plot(save=True, filename=fname.strip())
	except:
		print ("could not save {}".format(n.strip().replace('g141', 'G141')))
		
def get_cand_grism_ids():
	@numba.jit
	def format_name(name):
		n=name.split('/')[-1]
		n=n.split('.jpeg')[0]
		if n.startswith('par'): 
			n=n.replace('_', '-', 1)
		if not n.startswith('par'):
			n=n.replace('_', '-', 2)
		return n
	cands=glob.glob(SPECTRA_PATH+'/indices/*')
	lcands=[format_name(x) for x in cands]
	#save this into the new candidates files
	#df=	COMBINED_PHOTO_SPECTRO_DATA
	#print ((df[df.grism_id.isin(lcands)]).grism_id.values)
	#df[df.grism_id.isin(lcands)].to_pickle(LIBRARIES+'/candidates.pkl')
	pd.DataFrame(lcands).to_pickle(LIBRARIES+'/candidates.pkl')
	return lcands

def save_cands():
	cands.grism_id.apply(plot_name)

def save_again():
	#save again in the same file
	get_cand_grism_ids()
	df=pd.read_pickle(LIBRARIES+'/candidates.pkl')
	df.grism_id.apply(plot_name)

def look_at_all():
	import wisps
	df=wisps.datasets['rf_classified']
	print (len(df))
	df.grism_id.apply(plot_name)

