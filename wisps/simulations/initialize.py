import astropy.units as u
import numpy as np

CONSTANTS={'RSUN': 8.3*u.kpc.to(u.pc)*u.pc,
			'ZSUN': 2.7*u.kpc.to(u.pc)*u.pc,
			'N_0':0.39*u.pc**-3,
			'CCD_SIZE':1*u.arcmin**2,
			'SATURATION_LIMIT': {'J': 5}}