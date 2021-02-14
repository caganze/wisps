import numpy as np 
from astropy.coordinates import SkyCoord
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import wisps
import pandas as pd
import wisps.simulations as wispsim
import pickle
from tqdm import tqdm
#contants

POINTINGS= pd.read_pickle(wisps.OUTPUT_FILES+'/pointings_correctedf110.pkl')

DISTANCE_LIMITS=[]
SPGRID=wispsim.SPGRID
Rsun=8300.
Zsun=27.
HS=[250, 300, 350, 400, 450]
dist_arrays=pd.DataFrame.from_records([x.dist_limits for x in POINTINGS]).applymap(lambda x:np.vstack(x).astype(float))
DISTANCE_LIMITS={}
for s in SPGRID:
    DISTANCE_LIMITS[s]=dist_arrays[s].mean(axis=0)

#redefine magnitude limits by taking into account the scatter for each pointing 
#use these to compute volumes

#REDEFINED_MAG_LIMITS={'F110':    23.054573, 'F140':    23.822972, 'F160' :   23.367867}

#-------------------------------------------
def density_function(r, z, h=300.):
    
    """
    A custom juric density function that only uses numpy arrays for speed
    All units are in pc
    """
    l = 2600. # radial length scale of exponential thin disk 
    zpart=np.exp(-abs(z-Zsun)/h)
    rpart=np.exp(-(r-Rsun)/l)
    return zpart*rpart


def custom_volume(l,b,dmin, dmax, h):
    nsamp=1000
    ds = np.linspace(dmin,dmax,nsamp)
    rd=np.sqrt( (ds * np.cos( b ) )**2 + Rsun * (Rsun - 2 * ds * np.cos( b ) * np.cos( l ) ) )
    zd=Zsun+ ds * np.sin( b - np.arctan( Zsun / Rsun) )
    rh0=density_function(rd, zd,h=h )
    val=integrate.trapz(rh0*(ds**2), x=ds)
    return val

def interpolated_cdf(pnt, h):
	l, b= pnt.coord.galactic.l.radian, pnt.coord.galactic.b.radian
	d=np.concatenate([[0], np.logspace(-1, 4, int(1e3))])
	#print (d)
	cdfvals=np.array([custom_volume(l,b,0, dx, h) for dx in d])
	cdfvals= cdfvals/np.nanmax(cdfvals)
	return interp1d(d, cdfvals)

def draw_distance_with_cdf(pntname, dmin, dmax, nsample, h):
    #draw distances using inversion of the cumulative distribution 
	d=np.logspace(np.log10(dmin), np.log10(dmax), int(nsample))
	#print (d, dmin, dmax)
	cdfvals=(INTERPOLATED_CDFS[h][pntname])(d)
	return wisps.random_draw(d, cdfvals/np.nanmax(cdfvals), int(nsample))

INTERPOLATED_CDFS= {}
for h in HS:
            small_inter={}
            for p in POINTINGS:
                small_inter.update({p.name: interpolated_cdf(p, h)})
            INTERPOLATED_CDFS.update({h: small_inter })
#for k in INTERPOLATED_CDFS[150].keys(): print ((INTERPOLATED_CDFS[150][k])(3000))
#print (DISTANCE_LIMITS.flatten(axis=0))
#print ('above interpolation range? ', (INTERPOLATED_CDFS[150][POINTINGS[0].name])(3000))
#print ( draw_distance_with_cdf(POINTINGS[0].name, 0.5*(DISTANCE_LIMITS[17].flatten())[1], 2*(DISTANCE_LIMITS[17].flatten())[0], int(1e3), 150))
#fghj
def paralle_sample():
	#INTERPOLATED_CDFS= {}
	#for h in HS:
	#    small_inter={}
	#    for p in POINTINGS:
	#        small_inter.update({p.name: interpolated_cdf(p, h)})
	#    INTERPOLATED_CDFS.update({h: small_inter })
	DISTANCE_SAMPLES={}
	PNTAMES=[x.name for x in POINTINGS]
	print (PNTAMES)
	#print (DISTANCE_LIMITS)
	#for h in HS:
	#print (h)
	dis={}
	for h in HS:
		for s in tqdm(DISTANCE_LIMITS.keys()):
			dlts=np.array(DISTANCE_LIMITS[s]).flatten()
			fx= lambda x: draw_distance_with_cdf(x, 1., 2*dlts[0], int(1e4), h)
			with Pool() as pool:
				dx=pool.map(fx, PNTAMES)
				dis.update({s: dx})
				del dx
			DISTANCE_SAMPLES.update({h: dis})

		fl=wisps.OUTPUT_FILES+'/distance_samples{}'.format(h)
		with open(fl, 'wb') as file: pickle.dump(DISTANCE_SAMPLES[h],file, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ =='__main__':
	paralle_sample()
