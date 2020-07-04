#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: caganze

All  of the pre-processing i.e combing all photomteric and spectroscopic tables into a huge table

output: final cattalog of all the sources, their photometry and spectral indices

"""
from .initialize import *
import glob

import sys
from astropy.io import fits, ascii
from astropy.table import Table
import glob 
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from .initialize import *


def hst3d_phot_spec():
    """ combining all of HST-3D photometry catalogs and not ignoring columns
    """
    #photome
    cat_files=glob.glob('/users/caganze/*_3dhst*/Catalog/*.cat')
    cats=[]
    for c in cat_files:
        print (c)
        t=ascii.read(c).to_pandas()
        t['field']=c.split('_3dhst')[0].split('/')[-1]
        t['unique_id']=t['id'].apply(int).apply(str)+t['field'].apply(lambda x: x.lower())
        cats.append(t.reset_index(drop=True))

    phot=pd.concat(cats)
    #phot=ascii.read(cat_files[0]).to_pandas()
    phot=phot.rename(columns={'id':'phot_id'})

    print(phot.sample(n=10))


    #get grism_ids from the spectrum catalog
    hdu=(fits.open('/users/caganze/3dhst.v4.1.5.master.fits')[1])
    spec=Table(hdu.data).to_pandas()
    #spec=spec[spec.grism_id !='00000']

    #magnitudes are calculated to 25 zero-point
    def magnitude(flux):
        return 25.0-2.5*np.log10(flux)
    
    def mag_err(combined):
        #combined is a pandas table with flux and flux_error
    		if  np.isnan(combined['flux']):
    			return np.nan
    		else: return abs(0.434*2.5*combined['flux_error']/combined['flux'])
    #compute magnitudes
    for f in ['160', '140']:
        phot['F'+f+'_mag']= phot['f_F'+f+'W'].apply(magnitude)
        phot['Faper'+f+'_mag']= phot['faper_F'+f+'W'].apply(magnitude)
        cmbined1= pd.DataFrame()
        cmbined1['flux']= phot['f_F'+f+'W']
        cmbined1['flux_error']=phot['e_'+'F'+f+'W']
        
        cmbined2=pd.DataFrame()
        cmbined2['flux']=phot['faper_F'+f+'W']
        cmbined2['flux_error']=phot['eaper_F'+f+'W']
        
        phot['F'+f+'_mag_er']= cmbined1.apply(mag_err, axis=1)
        phot['Faper'+f+'_mag_er']=cmbined2.apply(mag_err, axis=1)

    

    #merge spec and phot on unique id, from photid, because photid is only unique within each field
    #this is done in order to obtain grism ids
    spec['unique_id']=spec['phot_id'].apply(int).apply(str)+spec['field'].apply(lambda x: x.lower())

    phot['unique_id']=phot['unique_id'].apply(lambda x:  x.replace('-', '').strip())
    spec['unique_id']=spec['unique_id'].apply(lambda x:  x.replace('-', '').strip())

    phot=phot.reset_index(drop=True)
    spec=spec.reset_index(drop=True)

    merged=pd.merge(spec, phot, on='unique_id', how='inner', validate='one_to_one')

    print (merged.shape)


    merged.to_csv(OUTPUT_FILES+'/hst3d_photometry_all.csv')
    
    
def wisp_phot_spec():
    """
    combine photomery and spectroscopy for wisps source"""
    
    def get_wisp_photometry_files(fld):
        """
        fld=folder
        """
        filterfiles_names= [140, 160, 110]
        
        files=[]
        for filterfile in filterfiles_names:
            	filepaths=REMOTE_FOLDER+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+'par'+str(fld)+'/*'+str(filterfile)+'*_cat.txt'
            	for f in  glob.glob(filepaths): files.append(f)
        print (files)  
        return files
    	
    def read_wisp_photometry_files(fld):
        phot=pd.DataFrame()
        files=get_wisp_photometry_files(fld)
        datas=[] #they're all the same table
        #print (files)
        if files != []:
            for f in  files:
                print (f)
                data=pd.read_table(f, skiprows=np.arange(0, 16), sep='\t')['#'].apply(reformat_phot_table)
                #print ('data {}, type {}'.format(len(data), type(data)))
                phot['F'+ f.split('_f')[-1].split('w_v6')[0]+'W']=data[12]
                phot['F'+ f.split('_f')[-1].split('w_v6')[0]+'W_ER']=data[13]
                phot['EXTRACTION_FLAG']=data[15]
                phot['star_flag']=data[14]
                datas.append(data)
            	
        #combine grism ids with corresponding photometry
        #choose one data that works
        if datas !=[]:
            print (datas)
            d=datas[0]
            #phot=phot.to_pandas()
            #d=d.to_pandas()
            phot['index']=phot.index
            d['index']=d.index
    
            joint_table=pd.merge(phot, d, on='index')
            #rename columns
            df=joint_table.rename(columns={0:'RA_DEC_NAME', 1:'NUMBER', 15: 'FLAGS'})
            
            return df
        
    def wisp_photometry():
        """ saves photometry for all Fields in the WISP SURVEY into 1 file  """
        folders= np.arange(1, 500)
        all_catalogs=[]
        #print "....saving mags ....."
        for fld in folders:
            phot_table=read_wisp_photometry_files(fld)
            all_catalogs.append(phot_table)
        all_catalogs=pd.concat(all_catalogs)
        return all_catalogs
    	
    def grism_id(row): return str(row.FIELD)+'-'+ str(row.NUMBER).zfill(5)
    
    def reformat_phot_table(row): return pd.Series(row.split())
    
    def returnra(radecname): return np.float(radecname.split('_')[1])
    
    def returndec(radecname): return np.float(radecname.split('_')[2])
    
    def returnfield(radecname): return 'Par'+str(radecname.split('_')[0])
    
    def ra_dec_field(radecname): return pd.Series({'RA':returnra(radecname), 'DEC':returndec(radecname), 'FIELD':returnfield(radecname)})
    
    wisp_phot=wisp_photometry()
    t=wisp_phot['RA_DEC_NAME'].apply(ra_dec_field)
    wisp_phot['RA']=t['RA']
    wisp_phot['DEC']=t['DEC']
    wisp_phot['FIELD']=t['FIELD']
    wisp_phot['grism_id']=wisp_phot.apply(grism_id, axis=1)
    
    wisp_phot.to_csv(OUTPUT_FILES+'/wisp_photometry.csv')
    print (wisp_phot)
    
    return wisp_phot