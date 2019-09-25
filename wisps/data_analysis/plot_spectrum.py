#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main plotting module for Source objects

@author: caganze
"""
from matplotlib import gridspec
from .initialize import *
from astropy.visualization import ZScaleInterval
import splat
#from .sepctrum_tools import plot_any_sp
from matplotlib import patches

def plot_source(sp, **kwargs):
    
    """
    Plotting routine for a source  (inlcudes photometry image)
    
    For a simpler plot, see Spectrum.plot
    
    """

    #flags
    cmap=kwargs.get('cmap', 'inferno')
    compare_to_std=kwargs.get('compare_to_std', True)
    save=kwargs.get('save', False)
    filt=kwargs.get('filter', 'F140W')
    
    #esthetiques
    xlim= kwargs.get('xlim', [1.1, 1.7])
    
    ylim=kwargs.get('ylim', [np.nanmin(sp.flux), np.nanmax(sp.flux)])
    xlabel=kwargs.get('xlabel','Wavelength ($\mu m$)')
    ylabel=kwargs.get('ylabel','Flux + c')
    if sp.survey=='hst3d':
    	xlim=[1.1, 1.65]
    
    #paths
    filename=kwargs.get('filename', OUTPUT_FIGURES+'/'+sp.name+'.pdf')
    
    #create the grid
    gs = gridspec.GridSpec(2, 3, height_ratios=(1, 3))
    fig=plt.figure(figsize=(8,6))
    ax1 = plt.subplot(gs[0, 0]) 
    ax2 = plt.subplot(gs[0, 1:3]) 
    ax3 = plt.subplot(gs[1, :]) 
    
    #remove markers from images
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.tick_params(axis='both', which='major', labelsize=15)
    
    
    l1,=ax3.step(sp.wave, sp.flux, color='#111111')
    l2,=ax3.plot(sp.wave, sp.noise, '#39CCCC')
    l4, =ax3.plot(sp.wave, sp.contamination, '#FF4136', linestyle='--')
    
    #print (np.nanmax(sp.flux[(1.25<sp.wave) & (sp.wave<1.6)]))
    #option to overplot the
   
    #get spectral type of the source
    if sp.spectral_type is None:
         sp.spectral_type=splat.classifyByStandard(sp.splat_spectrum,  comprng=[[1.1, 1.3], [1.3, 1.6]])[0]
    
    #collect plts
    plts=[l1, l2, l4]
    
    #compare to standards
    #if True:
    print ('plotting standard for {}'.format(sp.name))
    std=splat.getStandard(sp.spectral_type)
    std.normalize()
    chi, scale=splat.compareSpectra(sp.splat_spectrum, std,  comprng=[[1.1, 1.3], [1.3, 1.6]])
    #print ('scale ... {}'.format(scale))
    std.scale(scale)
    l3,=ax3.step(std.wave, std.flux, color='y')
    plts.append(l3)
    
    #ax3.set_xlim(xlim)
    ax3.set_xlabel(xlabel, fontsize=18)
    ax3.set_ylabel(ylabel, fontsize=18)
    
    #add the 2d spectrum
    v0, v1=ZScaleInterval().get_limits(sp.spectrum_image)
    ax2.imshow(sp.spectrum_image, vmin=v0, vmax=v1, cmap='Greys', aspect='auto')
    ax2.set_xlabel('G141', fontsize=15)
    
    #add the photo image

    image_data=None
    
    # mapping between filters and images
    image_data_dict={'F140W' : sp.photo_image.f140,
                'F160W': sp.photo_image.f160,
                'F110W': sp.photo_image.f110}
    try:
    	image_key_to_use=[ k for k in image_data_dict.keys() if image_data_dict[k]['grid'] is not None  ][0]
    except:
    	image_key_to_use='F140W'
    	
    
    mag_in_filter=np.round(sp.mags[image_key_to_use][0])
    
    	
    image_data=image_data_dict[image_key_to_use]
    vmin, vmax = ZScaleInterval().get_limits(image_data['data'])
    
    #white images
    if image_data['is_white']: 
    	ax1.imshow(image_data['data'])
    else:
    	ax1.pcolormesh(image_data['grid'][0], image_data['grid'][1], 
                   image_data['data'], cmap=cmap,
                   vmin=vmin, vmax=vmax, rasterized=True, alpha=1.0)
                   
    ax1.plot(image_data['center'][0], 
             image_data['center'][1], marker='+',c='#111111', ms=50)
             
    ax1.set_xlabel(image_key_to_use+'= '+str(mag_in_filter), fontsize=15)
    
    #print (np.nanmax(sp.flux[(1.25<sp.wave) & (sp.wave<1.6)]))
    #print (xlim)
    flux_for_lims=sp.flux[(xlim[0]<sp.wave) & (sp.wave<xlim[1])]
    flux_max=np.nanmax(flux_for_lims)
    ax3.set_xlim(xlim)
    ax3.set_xticks(np.arange(xlim[0], xlim[1], 0.01), minor=True)
    ax3.set_ylim([0.0, flux_max])

    bands=[[1.246, 1.295],[1.15, 1.20], [1.62,1.67], [1.56, 1.61], [1.38, 1.43]]
    bandlabels=['J-cont', '$H_2O-1$', '$CH_4$', 'H-cont', '$H_2O-2$']
    #if kwargs.get('overplot_bands', False):
    if kwargs.get('show_bands', True):
        for wrng, wlabel in zip(bands,  bandlabels):
            rect=patches.Rectangle((wrng[0], 0), wrng[1]-wrng[0], 1, angle=0.0, color='#DDDDDD')
            ax3.add_patch(rect)
            ax3.text(wrng[0],  flux_max-(flux_max/10.0),wlabel, {'fontsize':16} )

    
    lgd=ax3.legend(tuple(plts), (sp.shortname, 'Noise', 'contamination', '('+sp.spectral_type+') '+std.shortname), 
               loc=(1.01, 0.15), fontsize=15)
               
    #print(filename)
    
    if save: plt.savefig(filename,  bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    #plt.close()
    #fig.close()
    
    return fig