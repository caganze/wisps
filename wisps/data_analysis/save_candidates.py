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
	if os.path.isfile(fname) : pass
	else: plot(name, fname)
	
def plot(n, fname):
	try:
		s=Source(name=n.strip())
		s.plot(save=True, filename=fname.strip())
	except IndexError:
		print (n)

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
	df=	COMBINED_PHOTO_SPECTRO_DATA
	print (df[df.grism_id.isin(lcands)])
	df[df.grism_id.isin(lcands)].to_pickle(LIBRARIES+'/candidates.pkl')
	return lcands

def save_cands():
	cands.grism_id.apply(plot_name)

def look_at_all():
	import wisps
	df=wisps.datasets['rf_classified']
	print (len(df))
	df.grism_id.apply(plot_name)
