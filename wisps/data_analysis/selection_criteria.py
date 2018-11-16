#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This contains the main script I used to define selection criteria
"""

import matplotlib.patches as patches
import random as rd
from itertools import combinations
import pickle 
from functools import reduce
from .spectrum_tools import *
from wispshapes import Box
from .indices import*
from .spex_indices import *
from .initialize import *
from tqdm import tqdm


import pandas as pd
import statsmodels.nonparametric.kernel_density as kde

from matplotlib.path import Path
import matplotlib


#load a previous sample of potential brown dwarfs
df000=pd.read_csv(LIBRARIES+'/candidates.csv')
df100=pd.read_hdf(COMBINED_PHOTO_SPECTRO_FILE, key='all_phot_spec_data')
df200=df100[df100.grism_id.isin(df000.grism_id)]

#print (df200)
############################################
#format spex_sample ignore uncert
class Annotator(object):
    """
    Contains static method to manipulate index-index tables 
    """
    @staticmethod
    def  group_by_spt(df, **kwargs):
        
        """
        This is a static method that takes a table and an array of spectral type and 
        
        Args:
            df (pandas dataframe): a table of objects with a column of labelled spectral types

        Returns:
            returns the same table with spectral type ranges labelled
        """
        spt=kwargs.get('spt_label', 'Spts')
        #select by specral type range start spt=15
        df['spt_range']=''
        classes=['M7-L0', 'L0-L5', 'L5-T0','T0-T5','T5-T9']
        if kwargs.get('assign_middle', False):
            #assign the the range to the median spectral type
            classes=[20, 22, 27, 32, 37]

        if kwargs.get('assign_from_one', False):
            classes=[1, 2, 3, 4, 5]
        if not 'data_type' in df.columns:
            df['data_type']='templates'

        df['spt_range'].loc[(df[spt] >= 17.0 ) & (df[spt] <=20.0) & (df['data_type']== 'templates')]=classes[0]
        df['spt_range'].loc[(df[spt] >= 20.1 ) & (df[spt] <=25.0) & (df['data_type']== 'templates')]=classes[1]
        df['spt_range'].loc[(df[spt] >= 25.1 ) & (df[spt] <=30.0) & (df['data_type']== 'templates')]=classes[2]
        df['spt_range'].loc[(df[spt] >= 30.1 ) & (df[spt] <=35.0) & (df['data_type']== 'templates')]=classes[3]
        df['spt_range'].loc[(df[spt] >= 35.1 ) & (df[spt] <=40.0) & (df['data_type']== 'templates')]=classes[4]
        
        df['spt_range'].loc[ (df['data_type']== 'subdwarf')]='subdwarf'
        
        #print (df)
        if kwargs.get('add_subdwarfs', False):
            sds=kwargs.get('subdwarfs', None)
            #print ('adding subdwarfs')
            sds['spt_range']='subdwarf'
            df=pd.concat([df,sds],  ignore_index=True, join="inner")
        #print (df)
        return df

    @staticmethod
    def color_from_spts(spts, **kwargs):
        """
        Given spt (or a bunch of intergers, get colors
        spts must be arrays of numbers else, will try to change it to colors
        """
        if isinstance(spts[0], str):
            try:
                spts=[float(x) for x in spts]
            except:
                spts=[splat.typeToNum(x) for x in spts]
                
        cmap=kwargs.get('cmap', matplotlib.cm.YlOrBr)
        maxi= np.nanmax(spts)
        mini=np.nanmin(spts)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors=[]
        for c in spts:
                colors.append(mapper.to_rgba(c))
        
        return colors

    @staticmethod
    def reformat_table(df):
        """
        drop uncertainties in the indidces 
        """
        new_df=pd.DataFrame()
        for k in df.columns: 
            if isinstance(df[k].iloc[0], tuple):
                new_df[k]=np.array(np.apply_along_axis(list, 0, df[k].values))[:,0]
            else:
                new_df[k]=df[k].values
        return new_df
        
    #group_by_spt=staticmethod(group_by_spt)
    #color_from_spts=staticmethod(color_from_spts)

class Selector(object):
    """
    An object that helps me select using different logic
    """
    @staticmethod
    def select(subtype, index_spaces, data, **kwargs):
        """
        subtype: a string 'M7-L0' etc..
        index_spaces: a list of IndexSpace objects
        data must have columns consisent with the name of index-index spaces
        """
        _logic=kwargs.get('logic', 'and')
        if not isinstance(_logic, str) or _logic not in ['and', 'or']:
            raise ValueError(""" Logic must 'and' or 'or' """)
    
        selected=[]
        #print(data
        for idx in index_spaces:
                sls=idx.select(data, shapes_to_use=[subtype])
                sls=list(np.concatenate(sls))
                selected.append(list(sls))

        selected=list(selected)
        #print (selected)

        result=[]
        result_names=[]
        
        #print(selected)
        if _logic =='and':
            result_names=list(set.intersection(*map(set,selected)))
            #print (result_names)
        if _logic=='or':
            result_names=list(set().union(*selected))

        if len(result_names) !=0.0 :
            result=data[data.Names.isin(result_names)]

        #print (result_names)

        return result

class IndexSpace(object):
    """
    index-index space object
    
    Attributes:
    	name (str): (for example: 'CH_4/J-Cont H-cont/J-Cont')
    	shapes (list): rectangular boxes from shapes 
    	completeness (list): % of spex templates selected for each box
    	contamination (list): % of contaminants for each box
    	templates (pandas dataframe): a 3-column table (name, x_index, y_index) for all templates
    	subdwarfs (pandas dataframe): a 3-column table (name, x_index, y_index) for all subdwarfs
    	contaminants (pandas dataframe): a 3-column table (name, x_index, y_index) for all subdwarfs
    """
    def __init__(self, **kwargs):
        self._shapes=None       #all the shapes
        self._contaminants=None
        self._spex_templates=None
        self._contamination=None #contamination 
        self._completeness=None
        self.isBest= False #flag to specify if criterion will be used in the selectionprocess 
        self.xkey=kwargs.get('xkey', ' ')
        self.ykey=kwargs.get('ykey', ' ')
        self._spex_sample=None
        self._subdwarfs=None
        #self._name=None
    def __repr__(self):
        return 'index-index space of '+ self.name
        
    @property
    def name(self):
        return self.xkey+' '+ self.ykey
    
    @property
    def shapes(self):
        return self._shapes
    
    @property
    def contamination(self):
        return self._contamination
    
    @property
    def completeness(self):
        return self._completeness
    
    @property
    def contaminants(self):
        """
        A pandas dataframe  of contaminants
        """
        return self._contaminants
    
    @contaminants.setter
    def contaminants(self, new_conts):
        """
        Must be a pandas dataframe with columns
        """
        df=new_conts[[self.xkey, self.ykey]]
        df=df.dropna(how='any')
        df.columns=['x', 'y']
        #calculate the contamination based on the previous sample of potential brown dwarfs
        self._contaminants=new_conts[[self.xkey, self.ykey, 'Names']]
        #
        true_bds=(df200[[self.xkey, self.ykey]]).values.T
        cont= {}
        new_shapes=[]
        for s in self.shapes:
            s.datatype='contam'
            s.color=None
            s.data=np.array([df.x, df.y])
            #new definition of contamination
            #number that's selected that's actually brown dwarfs
            #print (true_bds)
            true_sl=len(s.select(true_bds).T)
            #number selected from all the trash
            trash_sl=len(s.select(s.data).T)
            #print (trash_sl)
            #calculate the contamination
            #print (true_sl, trash_sl)
            if trash_sl==0.0: cont[s.shape_name]=0.0
            else: cont[s.shape_name]=(trash_sl-true_sl)/(trash_sl)
            new_shapes.append(s)
        self._contamination=cont
        self._shapes=new_shapes
    
    @property
    def subdwarfs(self):
        return self._subdwarfs[[self.xkey, self.ykey, 'Names']]
    
    @subdwarfs.setter
    def subdwarfs(self, sds):
        #print (sds)
        sds=sds.dropna(how='any')
        #print (sds)
        sds=sds[[self.xkey, self.ykey, 'Names', 'Spts']]
        sds['data_type']='subdwarf'
        self._subdwarfs=sds
    	
    @property 
    def templates(self):
        """
        a pandas dataframe
        """
        return self._spex_sample
    
    @templates.setter
    def  templates(self, new_data):
        """
        Must pass pandas dataframe 
        should at least have xkey and ykey of the index-space

        Create selection shapes given for each spt range
        input: temalates columns should have at least
               x_key, y_key, 
        """
        new_data=new_data.dropna()
        #only keep columns that we need
        df= new_data[[self.xkey, self.ykey, 'Spts', 'Names']]
        #df.columns=['x', 'y', 'Spts', 'Names']
        df['data_type']='templates'
        self._spex_sample= new_data[[self.xkey, self.ykey, 'Spts', 'Names']]
        #rename columns
        annotated_df=None
        if not self._subdwarfs is None:
            annotated_df=Annotator.group_by_spt(df, add_subdwarfs=True, subdwarfs=self._subdwarfs)
        else:
            annotated_df=Annotator.group_by_spt(df)
            
        self._calc_completeness(annotated_df)
    
    #@classmethod   
    def _calc_completeness(self, annotated_df):
        """
        	This is how each box is defined after the user passes the templates property
        	Args:
            	annotated_df (pandas dataframe):must have a column 'spt_range' to differentatiate between M5-L0, ETC...
            Returns: 
            	None
        """
        grouped=annotated_df.groupby('spt_range')
        cpls={}
        new_shapes=[]
        for name, group in grouped:
            df=group.dropna()
            #print (df)
            if len(df) > 0.0:
                #print('name of the group ...{} length ... {}'.format(name, len(group)))
                to_use=df[[self.xkey, self.ykey]]
                to_use.columns=['x', 'y']
                xrng=[np.nanmin(np.array(to_use.x)), np.nanmax(np.array(to_use.x))]
                yrng=[np.nanmin(np.array(to_use.y)), np.nanmax(np.array(to_use.y))]
                box=Box()
                box.scatter_coeff=3.0
                box.alpha=1.0
                box.color=None
                box.shape_name=name
                box.xrange=xrng
                box.yrange=yrng
                box.data=np.array([to_use.x, to_use.y])
                new_shapes.append(box)
                cpls[name]=box.efficiency
                
        self._shapes=new_shapes
        self._completeness=cpls
        return 
    
    def select(self, df, **kwargs):
        """
        Method to select a bunch objects using this specific index.
        
        Args:
            df (pandas dataframe): a table that contains at least two columns of the index name for example: \
            if this index is named "one two", \
            then the table must have "one" and "two" as columns
        	**kwargs: you can specify which boxes to use in the selection e.g use the ["M5-L0", "T5-Y0"] boxes only
        	
        Returns:
            a list of the objects that fit in the boxes
        """
        #if not 
        df=df[[self.xkey, self.ykey, 'Names']]
        df.columns=['x', 'y', 'names']
        selected=[]
        #new_shapes=[]
        #can specify shapes to use (need to be passed as classes)
        input_boxes= kwargs.get('shapes_to_use', self.shapes)
        
        #if the user passed a string, retrieve the name of the box
        use=input_boxes
        
        if all(isinstance(bx, str) for bx in use):
        	use=[x for x in self.shapes if x.shape_name in input_boxes]
    

        if kwargs.get('table', False):
            return 

        if not kwargs.get('table', False):
            for s in use:
                rows=s.select(df[['x', 'y']]).index
                sels=list(np.unique(df['names'].loc[rows].tolist()))
                selected.append(sels)

            return list(selected)
    

    def plot(self, **kwargs):
        """
        Plotting function for an index-index space
    
        """
        dict1={'data':[self.templates[self.xkey].tolist(), self.templates[self.ykey].tolist()], 
               'name':'Templates', \
                'color':'#FFDC00', 
                'marker':'D', 'ms':5, 'cmap':'YlOrBr', \
                  'alpha':1, 'edgecolor':'none', 'linewidths':3}
        
        dict3={'data':[self.contaminants[self.xkey].tolist(),self.contaminants[self.ykey].tolist()], 
                          'name':'Contaminants', \
                       'color':'#FFDC00', 'marker':'+', 'ms':5, 'cmap':'YlOrBr', \
                       'alpha':0.3, 'edgecolor':'#111111', 'linewidths':3}
        
        ddicts=[ dict1]
        
        if kwargs.get('show_subdwarfs', True):
            dict2={'data':[self.subdwarfs[self.xkey].tolist(),self.subdwarfs[self.ykey].tolist()], 
                      'name':'Subdwarfs', \
                   'color':'#FFDC00', 'marker':'D', 'ms':5, 'cmap':'YlOrBr', \
                   'alpha':1, 'edgecolor':'k', 'linewidths':3}
            ddicts.append(dict2)
        
        datadicts=kwargs.get('data_dicts', ddicts)

        fig=plt.figure(figsize=kwargs.get('figsize', (8,8)))
        ax1 = fig.add_subplot(111)
        #ax1.set_xscale("log")
        #ax1.set_yscale("log")
        #_#sp_sample=self
         #print (templates_data)
        templates_data =[d['data'] for d in datadicts if d['name']=='Templates']
        if templates_data != []:
            xmean=np.mean(templates_data[0][0])
            xstd=np.std(templates_data[0][0])
            ymean=np.mean(templates_data[0][1])
            ystd=np.std(templates_data[0][1])
            xmin=xmean-5.0*xstd
            ymin=ymean-5.0*ystd
            if xmin<0.0:
                xmin=0.0
            if ymin<0.0:
                ymin=0.0
            xlims=[xmin, xmean+5.0*xstd]
            ylims=[ymin, ymean+5.0*ystd]
            
        else:
            xlims=self.shapes[-1].xrange
            ylims= self.shapes[-1].yrange
    
            
        for d in datadicts:
            ax1.scatter(d['data'][0],d['data'][1],  c=d['color'], marker=d['marker'],
                        s=d['ms'], cmap=d['cmap'], label=d['name'], alpha=d['alpha'], edgecolor=d['edgecolor'], 
                        linewidths=d['linewidths'])
                        
        x=np.array(dict3['data'][0])
        y=np.array(dict3['data'][1])
        
        small_df=pd.DataFrame([x, y]).transpose()
        small_df.columns=['x', 'y']
        small_df=small_df.replace([np.inf, -np.inf], np.nan)
        small_df=small_df.dropna(how='any')
        
        #only show things in range of the plot
        small_df=small_df[(small_df.x.between(xlims[0],xlims[1]))& (small_df.y.between(ylims[0], ylims[1] ))]
        
        xd=small_df.x.as_matrix()  #rawr xd :) lol
        yd=small_df.y.as_matrix()
        
        # print (xd[np.isnan(xd)], yd[np.isnan(yd)])
        #print (xd, yd)
        #heatmap, xedges, yedges = np.histogram2d(xd, yd, bins=10000)
        #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        
        #ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
        hist2d=ax1.hist2d(x=xd, y=yd, cmap=kwargs.get('cmap', MYCOLORMAP), bins=10, alpha=1.0)
        
        #ax1.hist2d(x=xd, y=yd, cmap='Reds')
        cbar = fig.colorbar(hist2d[3],  orientation='horizontal')
        df_=pd.DataFrame()
        df_['x']=d['data'][0]
        df_['y']=d['data'][1]
        for s in self.shapes:
            #print ('plotting ....', s.shapetype)
            s.color='none'
            s.alpha=0.3
            if s.shape_name in kwargs.get('highlight', [None]): s.plot( ax=ax1, only_shape=True, highlight=True)
            else: s.plot( ax=ax1, only_shape=True)
            #s.color='none'
            #print (""" itsssssssss """,  kwargs.get('highlight', None))
            #print ('name', s.shape_name, 'color', s.color, 'linewidth', s.linewidth)
            
        plt.xlabel('$'+str(self.name.split(' ')[0])+'$', fontsize=18)
        plt.ylabel('$'+str(self.name.split(' ')[1])+'$', fontsize=18)
        
      
        
        if kwargs.get('log_scale', False):
            plt.xscale('log', nonposy='clip')
            plt.yscale('log', nonposy='clip')
            
        filename=kwargs.get('filename', 'none')
        #set limits of the plts from templates 
        plt.xlim(kwargs.get('xlim',  xlims))
        plt.ylim(kwargs.get('ylim', ylims))
        plt.legend()
        plt.show()
        plt.close()
        filenm=kwargs.get('filename', 
        OUTPUT_FIGURES+'/indices/index_plot_'+self.name.replace('/','_').replace('-', '_').replace(' ', '_')+'.pdf')
        if kwargs.get('save', True):
            fig.savefig(filenm, dpi=100, bbox_inches='tight')
            
        return

def crts_from_file(**kwargs):
    """
    loads saved selection criteria
    """
    filename=kwargs.get('filename',OUTPUT_FILES+'/id_id_spaces_cpl_all_shapes.pkl')
    return pd.read_pickle(filename)


def save_criteria(**kwargs):
    """
    creates selection criteria
    table of selection criteria: ranges, completeness
    """
    #load templates (will return spectral type and 10 indices for each object
    #completeness =kwargs.get('completeness', 0.9)
    templates=spex_sample_ids(stype='spex_sample',  from_file=True)
    tpl_ids=pd.DataFrame([x for x in templates['Indices']])
    #templates['data_type']= 'templates'
    
    subdwarfs=spex_sample_ids(stype='sd',  from_file=True)
    sd_ids=pd.DataFrame([x for x in subdwarfs['Indices']])
    #subdwarfs['data_type']= 'subdwarf'

    #print(subdwarfs)
    
    for k in tpl_ids.keys(): 
        templates[k]=np.array(np.apply_along_axis(list, 0, tpl_ids[k].values))[:,0]
        subdwarfs[k]=np.array(np.apply_along_axis(list, 0, sd_ids[k].values))[:,0]
    
    #contaminants, should be using file that was generated after signal to noise cut
    #data_file=kwargs.get('cont_file', OUTPUT_FILES+'//new_contaminants.pkl')
    #conts=pd.read_pickle(data_file)
    conts=kwargs.get('conts', COMBINED_PHOTO_SPECTRO_DATA)
    #print(conts)
    conts=conts.rename(columns={'grism_id': 'Names'})
    keys=tpl_ids.columns
    index_spaces=[]
    for x_key, y_key in  tqdm(list(combinations(keys,2))):
        idspace=IndexSpace(xkey=x_key, ykey=y_key)
        #print (idspace.name)
        #pass subdwarfs first 
        idspace.subdwarfs=subdwarfs
        idspace.templates=templates
        idspace.contaminants=conts
        index_spaces.append(idspace)
        
    #save all 45 id-id spaces in a file
    names=[x.name for x in index_spaces]
    idx_space_dict=dict(zip(*[names, index_spaces]))
    output = open( OUTPUT_FILES+'//id_id_spaces_cpl_all_shapes.pkl', 'wb')
    pickle.dump(idx_space_dict, output)
    output.close()
    
    return index_spaces

def plot_cont_compl(**kwargs):

		"""
		plottting the contamination and completeness heatmaps
		"""
		cmap=kwargs.get('cmap', MYCOLORMAP)
		
		crts=crts_from_file()
		conts=pd.DataFrame([ x.contamination for x in crts])
		compls=pd.DataFrame([ x.completeness for x in crts])
		conts['index-space']=[x.name for x in crts]
		compls['index-space']=[x.name for x in crts]

		conts.index=['idx'+str(i) for i in range(0, len(conts['index-space']))]
		new_conts=conts.sort_values(by=list(conts.columns), ascending=True).drop(labels='index-space', axis=1)
		compls.index=['idx'+str(i) for i in range(0, len(compls['index-space']))]
		new_compls=compls.sort_values(by=list(compls.columns), ascending=True).drop(labels='index-space', axis=1)
		
		fig, (ax1, ax2)=plt.subplots(1, 2, figsize=kwargs.get('figsize',(10, 6)), sharex=True, sharey=True)
		seaborn.heatmap(new_conts, cmap=cmap, ax=ax1)
		seaborn.heatmap(new_compls,  ax=ax2, cmap=cmap)
		
		ax2.set_title('completeness', fontsize=14)
		ax1.set_title('contamination', fontsize=14)
		
		fig.savefig( OUTPUT_FIGURES+'/completeness_contamination.pdf', bbox_inches='tight')
		
		return 
		
def pick_by_contamination(input_dict):
	"""
	Given a contamination cut and
	input dictionary should 
	"""


if __name__ =="__main__":
    crtFromfile()
   
