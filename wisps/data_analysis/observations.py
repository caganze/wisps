# -*- coding: utf-8 -*-
"""
This combines photometric logs, 
i don't remember if I need but i'll keep it just in case
"""

from .initialize import *
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
import glob

def get_paths():
    """
    gets paths for all pointings in both surveys
    """
   
    #a script that loops through the folder and reads in the fits images to extract
    path=REMOTE_FOLDER
    hst_paths=[path+p for  p in ['aegis', 'goods', 'cosmos', 'uds']]
    wisp_path=path+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'
    ppaths=[]
    for p in ['aegis',  'cosmos', 'uds', 'goods']:
        ppaths.extend(glob.glob(path+p+'/*/*F1*0W_drz_sci.fits'))
    ppaths.extend(glob.glob(wisp_path+'/par*/hlsp_wisp_hst_wfc3_*-80mas_f*w_v6.2_drz.fits'))
    ppaths.extend(glob.glob(path+'goods*'+'*F1*0W_drz_sci.fits'))

    return ppaths

def create_logs():

    """
    Gets useful table columns 

    """
    hst_data=pd.DataFrame()
    ras=[]
    decs=[]
    obs_ts=[]
    exp_ts=[]
    fields=[]
    ppaths=get_paths()
    for ppath in ppaths:
        print (ppath)
        data=fits.open(ppath)[0]
        if ppath.split('/')[4]=='wisps':
            fields.append('wisps-'+ppath.split('/Par')[1].split('_')[0])
            #print (ppath.split('/')[-4].replace('_final_V5.0', ' '))
            #print (data.header['DATE-OBS'])
            obs_ts.append(data.header['DATE-OBS'])
        else:
            fields.append(ppath.split('/')[-2])
            #convert times from julian dates
            ts=data.header['EXPSTART']
            t= Time(ts, format='mjd')
            #print (ppath.split('/')[-2])
            #print (t)
            obs_ts.append(t.iso.split()[0])
        #hjk
        exp_ts.append(data.header['EXPTIME'])
        ras.append(data.header['RA_TARG'])
        decs.append(data.header['DEC_TARG'])
        
        
        #if p=='wisps':
    print (len(ras), len(decs), len(obs_ts), len(fields))
    #hst_data['L_MAG']=
    #hst_data['NIMAGING']=
    hst_data['RA (deg)']=ras
    hst_data['DEC(deg)']=decs
    #galactic coordinates
    c_icrs = SkyCoord(ra=ras*u.degree, dec=decs*u.degree, frame='icrs')
    hst_data['l (deg)']=c_icrs.galactic.l
    hst_data['b (deg)']=c_icrs.galactic.b
    hst_data['EXPOSURE (s)']=np.round(exp_ts)
    hst_data['OBSERVATION DATE (UT)']=obs_ts
    hst_data['POINTING']=fields

    hst_data.to_latex(OUTPUT_FILES+'/observation_log.tex', index=False)
    hst_data.to_csv(OUTPUT_FILES+'/observation_log.csv')

    print (hst_data)
    return hst_data

if __name__ == '__main__':
	create_logs()