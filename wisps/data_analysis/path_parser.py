# -*- coding: utf-8 -*-

"""

After the introduction of version 6.2, all wisp data and hst-3d are now on MAST

3D-HST has not added any new data nor changed their directory structure, 
but that's not the case for WISP

Aim: parse new directories to make them compatible with v5.0
"""

import os 
import glob
from ..utils import memoize_func

REMOTE_FOLDER=os.environ['WISP_SURVEY_DATA']

@memoize_func
def get_image_path(name, spectrum_path):
    #print (name)
    ##returns the image path without going through the whole thing again
    if name.startswith('Par') or name.startswith('par') or name.startswith('hlsp'):
        survey='wisps'

    elif name.startswith('goo') or  name.startswith('ud') or  name.startswith('aeg')  or name.startswith('cos'):
        survey='hst3d'

    if  survey=='wisps':
        folder=name.split('wfc3_')[-1].split('wfc3_')[-1].split('-')[0]
        if '_wfc3' in name:
            name=(name.split('wfc3_')[-1]).split('_g141')[0]
        #print (name)
        #print (REMOTE_FOLDER+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+folder+'*/2dstamp/hlsp_wisp_hst_wfc3*'+name+'*stamp2d.fits')
        stamp_image_path=glob.glob(REMOTE_FOLDER+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+folder+'*/2dstamp/hlsp_wisp_hst_wfc3*'+name+'*stamp2d.fits')[0]

    if survey=='hst3d':
        #print (spectrum_path.split('/1D/ASCII/')[0]+'/2D/'+'FITS/'+name.split('1D')[0]+'*2D.fits')
        stamp_image_path=glob.glob(spectrum_path.split('/1D/ASCII/')[0]+'/2D/'+'FITS/'+name.split('1D')[0]+'*2D.fits')[0]
        #print ('stamp image',stamp_image_path )
    #print (survey, spectrum_path, stamp_image_path)
    return survey, stamp_image_path


@memoize_func
def parse_path(name, version):
    """
    Parse a filename and retrieve all the survey info at once
    """
    survey=None
    spectrum_path=None
    stamp_image_path=None
    if name.startswith('Par') or name.startswith('par') or name.startswith('hlsp'): 
        survey='wisps'

    elif name.startswith('goo') or  name.startswith('ud') or  name.startswith('aeg')  or name.startswith('cos'):
        survey='hst3d'
    else:
        survey=None

    if  survey=='wisps':
        spectrum_path=_run_search(name)
        folder=name.split('wfc3_')[-1].split('wfc3_')[-1].split('-')[0]
        name=name.split('_wfc3_')[-1].split('a_g102')[0]
        stamp_image_path=glob.glob(REMOTE_FOLDER+'/wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+folder+'*/2dstamp/hlsp_wisp_hst_wfc3*'+name+'*stamp2d.fits')[0]
    if survey=='hst3d':
        spectrum_path=_run_search(name)
        #print (spectrum_path.split('/1D/ASCII/')[0]+'/2D/'+'FITS/'+name.split('1D')[0]+'*2D.fits')
        stamp_image_path=glob.glob(spectrum_path.split('/1D/ASCII/')[0]+'/2D/'+'FITS/'+name.split('1D')[0]+'*2D.fits')[0]
        #print ('stamp image',stamp_image_path )
    #print (survey, spectrum_path, stamp_image_path)
    #blah

    return survey, spectrum_path, stamp_image_path


@memoize_func
def _run_search(name):
    #internal function used to search path given spectrum name
    path=''
    prefix= name[:3]
    if name.startswith('Par') or name.startswith('par') or name.startswith('hlsp'):
        try:
         	#search version 6
            if name.endswith('.dat'):
                n=name.split('.dat')[0]
                folder=name.split('wfc3_')[-1].split('wfc3_')[-1].split('-')[0]
            else:
                folder=name.split('-')[0]
                n=name
            path=REMOTE_FOLDER+'wisps/archive.stsci.edu/missions/hlsp/wisp/v6.2/'+folder+'/1dspectra/*'+n+'*a_g141_*'
            #print (path)
            path=glob.glob(path)[0]

        except:
         	#search version 5
            folder=name.split('_')[0]
            path=REMOTE_FOLDER+'wisps/'+folder+'*/Spectra/*'+name+'.dat'
            #print (path)
            path=glob.glob(path)[0]

    if prefix in ['aeg', 'cos', 'uds', 'goo']:
     try:
        #if '-' in name : 
        #if '_' in name : syls= (name.split('_'))
        syls= (name.split('-'))
        str_= REMOTE_FOLDER+'*'+prefix+'*'+'/*'+prefix+ '*'+syls[1]+'*'+'/1D/ASCII/'+prefix+'*'+ syls[1]+ '*'+syls[2]+'*'
        
        path=glob.glob(str_)[0]
     except:
        path=''
        #print (path)
    #print(path)
    return path

@memoize_func
def return_path(name):
    #print(name)
    if type(name) is list:
        paths=[]
        for p in name:
            paths.append( _run_search(p))
        return paths
    if type(name) is str:
        return _run_search(name)
    
@memoize_func
def return_spectrum_name(path):
    """ returns name given path in the wisp folder"""
    name=''
    try:
     	name= path.split('.dat')[0].split('/')[-1]
    except:
     	name=path.split('.ascii')[0].split('/')[-1].split('.')[0]

    return name