#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: caganze

All  of the pre-processing i.e combing all photomteric and spectroscopic tables into a huge table

output: final cattalog of all the sources, their photometry and spectral indices

"""
from .initialize import *

import sys
from astropy.io import fits, ascii
from astropy.table import Table
import glob 
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from .initialize import *


def hst3d_phot_spec():
    """ combining all of HST-3D photometry and spectroscopy
    """
    #readin-in the files
    f1=REMOTE_PHOT_CATALOGS+'/goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat'
    f2=REMOTE_PHOT_CATALOGS+'goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat'
    f3=REMOTE_PHOT_CATALOGS+'aegis_3dhst.v4.1.cats/Catalog/aegis_3dhst.v4.1.cat'
    f4=REMOTE_PHOT_CATALOGS+'cosmos_3dhst.v4.1.cats/Catalog/cosmos_3dhst.v4.1.cat'
    f5=REMOTE_PHOT_CATALOGS+'uds_3dhst.v4.2.cats/Catalog/uds_3dhst.v4.2.cat'
    
    goodsn=ascii.read(f1).to_pandas()
    goodss=ascii.read(f2).to_pandas()
    aegis=ascii.read(f3).to_pandas()
    cosmos=ascii.read(f4).to_pandas()
    uds=ascii.read(f5).to_pandas()
    
    goodsn['field']='goodsn'
    goodss['field']='goodss'
    aegis['field']='aegis'
    cosmos['field']='cosmos'
    uds['field']='uds'
    
    hst3d_phot=pd.concat([goodsn, goodss,
                     aegis, cosmos,
                     uds])
    
    #read in the spectroscopy
    hdu=(fits.open(REMOTE_PHOT_CATALOGS+'3dhst.v4.1.5.master.fits')[1])
    hst3d_specp=Table(hdu.data).to_pandas()
    print (hst3d_specp.shape, hst3d_phot.shape)
    
    hst3d_phot=hst3d_phot.rename(columns={'id': 'phot_id'})
    hst3d_specp['phot_id']= hst3d_specp['phot_id'].apply(int)
    hst3d_phot['phot_id']=  hst3d_phot['phot_id'].apply(int)
    
    hst3d_specp['unique_id']= hst3d_specp['field'].astype(str)+hst3d_specp['phot_id'].astype(int).astype(str)
    hst3d_phot['unique_id']=hst3d_phot['field'].astype(str)+hst3d_phot['phot_id'].astype(int).astype(str)
    
    hst3d_specp['unique_id']= hst3d_specp['unique_id'].apply(str.lower)
    hst3d_phot['unique_id']= hst3d_phot['unique_id'].apply(str.lower)
    
    combined=pd.merge(hst3d_phot, hst3d_specp, on='unique_id', how='inner')
    hst_3d=combined.reset_index(drop=True)
    
    #remove some columns
    #define some specific functions
    
    #magnitudes are calculated to 25 zeo-point
    def magnitude(flux):
        return 25.0-2.5*np.log10(flux)
    
    def mag_err(combined):
        #combined is a pandas table with flux and flux_error
    		if  np.isnan(combined['flux']):
    			return np.nan
    		else: return abs(0.434*2.5*combined['flux_error']/combined['flux'])
    #compute magnitudes
    for f in ['160', '140']:
        hst_3d['F'+f+'_mag']= hst_3d['f_F'+f+'W'].apply(magnitude)
        hst_3d['Faper'+f+'_mag']= hst_3d['faper_F'+f+'W'].apply(magnitude)
        cmbined1= pd.DataFrame()
        cmbined1['flux']= hst_3d['f_F'+f+'W']
        cmbined1['flux_error']=hst_3d['e_'+'F'+f+'W']
        
        cmbined2=pd.DataFrame()
        cmbined2['flux']=hst_3d['faper_F'+f+'W']
        cmbined2['flux_error']=hst_3d['eaper_F'+f+'W']
        
        hst_3d['F'+f+'_mag_er']= cmbined1.apply(mag_err, axis=1)
        hst_3d['Faper'+f+'_mag_er']=cmbined2.apply(mag_err, axis=1)
    
    #format pointing names
    def get_pointing_name(grism_id):
        """
        return pointing name givem grism_id
        """
        if str(grism_id) != '00000':
            return str(grism_id).split('-G141')[0]
        else: return grism_id
        
    hst_3d['pointing_name']=hst_3d['grism_id'].apply(get_pointing_name)
    	#only select certain columns 
    important_columns=['phot_id_x', 'grism_id', 'field_x', 'ra_x', 'dec_x','faper_F140W', 'eaper_F140W',\
                   'faper_F160W', 'eaper_F160W','f_F140W', 'e_F140W', 'f_F160W',\
                   'e_F160W', 'F160_mag', 'Faper160_mag', 'F140_mag', 'Faper140_mag', \
                   'F160_mag_er', 'Faper160_mag_er', 'F140_mag_er', 'Faper140_mag_er', 
                   'jh_mag', 'flags', 'pointing_name', 'use_phot_x', 'f_cover', 'f_flagged', 'f_negative']
    
    hst_3d[important_columns].to_csv(OUTPUT_FILES+'/hst3d_photometry_all.csv')
    
    
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
            	filepaths=REMOTE_FOLDER+'/wisps/'+'Par'+str(fld)+'_final*/DATA/DIRECT_GRISM/fin_F'+str(filterfile)+'.cat'
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
                phot['NIMCOS_'+str(f.split('_')[-1].split('.')[0])+'W']=data[12]
                phot['NIMCOS_'+str(f.split('_')[-1].split('.')[0])+'W_ER']=data[13]
                phot['EXTRACTION_FLAG']=data[15]
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
    	
    def grism_id(row): return str(row.FIELD)+'_BEAM_'+ str(row.NUMBER)+ 'A'
    
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