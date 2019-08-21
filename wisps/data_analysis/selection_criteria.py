
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
from ..data_sets import datasets


import pandas as pd
import statsmodels.nonparametric.kernel_density as kde

from matplotlib.path import Path
import matplotlib


#load a previous sample of potential brown dwarfs

df200=datasets['candidates']
mjdf=datasets['manjavacas']
scndf=datasets['schneider']
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

        spt_label=keyword for spt column
        """
        spt=kwargs.get('spt_label', 'Spts')
        #select by specral type range start spt=15
        df['spt_range']=''
        classes=['trash', 'M7-L0', 'L0-L5', 'L5-T0','T0-T5','T5-T9']
        if kwargs.get('assign_middle', False):
            #assign the the range to the median spectral type
            classes=[20, 22, 27, 32, 37]

        if kwargs.get('assign_number', False):
            classes=[0, 1, 2, 3, 4, 5]
        if not 'data_type' in df.columns:
            df['data_type']='templates'

        df['spt_range'].loc[(df[spt] < 17.0 ) & (df['data_type']== 'templates')]=classes[0]
        df['spt_range'].loc[(df[spt] >= 17.0 ) & (df[spt] <=20.0) & (df['data_type']== 'templates')]=classes[1]
        df['spt_range'].loc[(df[spt] >= 20.1 ) & (df[spt] <=25.0) & (df['data_type']== 'templates')]=classes[2]
        df['spt_range'].loc[(df[spt] >= 25.1 ) & (df[spt] <=30.0) & (df['data_type']== 'templates')]=classes[3]
        df['spt_range'].loc[(df[spt] >= 30.1 ) & (df[spt] <=35.0) & (df['data_type']== 'templates')]=classes[4]
        df['spt_range'].loc[(df[spt] >= 35.1 ) & (df[spt] <=40.0) & (df['data_type']== 'templates')]=classes[5]
        
        df['spt_range'].loc[ (df['data_type']== 'subdwarf')]='subdwarf'
        
        #print (df)
        if kwargs.get('add_subdwarfs', False):
            sds=kwargs.get('subdwarfs', None)
            #print ('adding subdwarfs')
            sds['spt_range']='subdwarfs'
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
        self._shapes=[]      #all the shapes
        self._contaminants=None
        self._spex_templates=None
        self._contamination={} #contamination 
        self._completeness={}
        self.isBest= False #flag to specify if criterion will be used in the selectionprocess 
        self.xkey=kwargs.get('xkey', ' ')
        self.ykey=kwargs.get('ykey', ' ')
        self._spex_sample=None
        self._subdwarfs=None
        self._false_negative=None
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
        df=Annotator.reformat_table(new_conts[[self.xkey, self.ykey]])
        df=df.dropna(how='any')
        df.columns=['x', 'y']
        #calculate the contamination based on the previous sample of potential brown dwarfs
        self._contaminants=new_conts[[self.xkey, self.ykey, 'Names']]
        #
        #true_bds=Annotator.reformat_table(df200[[self.xkey, self.ykey]].applymap(eval)).values.T
        #print (true_bds)
        cont= {}
        fn={}
        new_shapes=[]
        for s in self.shapes:
            s.datatype='contam'
            s.color=None
            s.data=np.array([df.x, df.y])
        
            slctd=s.select(s.data)

            if len(s.data.T)==0.0:
                cont[s.shape_name]=0.0
            else:
                cont[s.shape_name]=len(slctd.T)/len(s.data.T)

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
       # print (self.xkey, self.ykey, new_data.columns)
        df['data_type']='templates'
        self._spex_sample= new_data[[self.xkey, self.ykey, 'Spts', 'Names']]

        #print (self._spex_sample.shape)
        annotated_df=None
        if not self._subdwarfs is None:
            annotated_df=Annotator.group_by_spt(df, add_subdwarfs=True, subdwarfs=self._subdwarfs)
        else:
            annotated_df=Annotator.group_by_spt(df)
        
        self._calc_completeness(annotated_df)
    
    #@classmethod   
    def add_box(self, df, name, color, coeff):
        """
        Adds a box to the selection criteria
        """
        #print (df)
        #reformat the data
        x=np.array([*list(df.x.values)])
        y=np.array([*list(df.y.values)])
        ddf=pd.DataFrame([x[:,0], y[:,0]]).transpose().dropna()
        #create a box
        box=Box()
        if name.lower().startswith('l') or name.lower().startswith('y') or name.lower().startswith('m'): 
            box=Box(shapetype='rectangle')
        box.scatter_coeff=coeff
        box.alpha=.1
        box.color=color
        box.shape_name=name
        box.edgecolor='#2ECC40'
        #print (ddf.values.T, name)
        box.data=np.array(ddf.values.T)
        #add this to the existing 
        self._shapes.append(box)
        self._completeness[name]=box.efficiency

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
            #print (df.shape, len(df))
            if len(df) > 0.0:
                #print('name of the group ...{} length ... {}'.format(name, len(group)))
                to_use=df[[self.xkey, self.ykey]]
                to_use.columns=['x', 'y']
                self.add_box(to_use, name, '#0074D9', 3.0)


        #add an extra box of late ts from manjavacas et al
        #print (self.completeness)
        mdf=mjdf[[self.xkey, self.ykey, 'spt']]
        mdf.columns=['x', 'y', 'spt']


        #add schneider objects
        sdf= scndf[[self.xkey, self.ykey, 'spt']]
        sdf.columns=['x', 'y', 'spt']

        ydwarfs=mdf[mdf['spt'].apply(lambda x: splat.typeToNum(x))>37].append(sdf)

        self.add_box(ydwarfs, 'Y dwarfs', '#0074D9', 3.0)

        #print (self.completeness)

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

    def new_plot_best(self, box_label, **kwargs):
        #newplotiing function only looking at one box
        bs=self.shapes
        bx=[x for x in bs if x.shape_name==box_label][0]
        print (box_label)

        spex_df=Annotator.reformat_table(datasets['spex']).reset_index(drop=True)
        manj=Annotator.reformat_table(datasets['manjavacas']).reset_index(drop=True)
        schn=Annotator.reformat_table(datasets['schneider']).reset_index(drop=True)
        subdwarfs=Annotator.reformat_table(self._subdwarfs).reset_index(drop=True)
        cands=Annotator.reformat_table(datasets['candidates']).reset_index(drop=True)

        manj['Spts']=manj.spt.apply(splat.typeToNum)
        schn['Spts']=schn.spt.apply(splat.typeToNum)
        cands['Spts']=cands.spt.apply(splat.typeToNum)

        spex_df=Annotator.group_by_spt(spex_df, spt_label='Spts')
        schn=Annotator.group_by_spt(schn, spt_label='Spts')
        manj=Annotator.group_by_spt(manj, spt_label='Spts')
        cands=Annotator.group_by_spt(cands, spt_label='Spts')

        ydwarfs=manj[manj['Spts'].apply(lambda x: x>37)]

        if 'ax' in kwargs:
            ax= kwargs.get('ax', None)
        else:
            fig=plt.figure(figsize=kwargs.get('figsize', (8,8)))
            ax=fig.add_subplot(111)

        conts=self.contaminants
        xkey, ykey=self.xkey,self.ykey

        ax.scatter((self.contaminants[xkey]).apply(float).round(2), (self.contaminants[ykey]).apply(float).round(2),  marker='o',  facecolors='none',  edgecolors='#AAAAAA', label='Contaminants')

        if box_label.lower()=='y dwarfs':
            ax.scatter(ydwarfs[xkey], ydwarfs[ykey], label='Y dwarfs')
        if box_label.lower() =='subdwarfs':
            ax.scatter(subdwarfs[xkey].apply(float), subdwarfs[ykey].apply(float), label='subdwarfs')
        if (box_label.lower != 'y dwarfs') and (box_label.lower != 'subdwarfs'):
            s_spex=spex_df[spex_df.spt_range==box_label]
            s_manj=manj[manj.spt_range==box_label]
            s_schn=schn[schn.spt_range==box_label]
            s_cand=cands[cands.spt_range==box_label]
            #print (s_cand)

            ax.scatter(s_spex[xkey], s_spex[ykey], s=5, label='SpeX')
            #ax.scatter(s_manj[xkey], s_manj[ykey],  marker='P', facecolors='none', edgecolors='#FF851B', label='Manjavacas')
            #ax.scatter(s_schn[xkey],  s_schn[ykey],  marker='^', facecolors='none', edgecolors='#B10DC9', label='Schneider')
            ax.scatter((s_cand[xkey]).apply(float).round(2), (s_cand[ykey]).apply(float).round(2), marker='x', facecolors='#111111', edgecolors='#2ECC40', label='candidates')


        bx.plot( ax=ax, only_shape=True, highlight=False)

        filename=kwargs.get('filename', 'none')
        #set limits of the plts from templates 
        ax.set_xlim(kwargs.get('xlim',  [0., 10.]))
        ax.set_ylim(kwargs.get('ylim', [0., 10.]))

        #indices that use the continuum have ranges that are too high, logscale this?

        ax.set_xlabel(r'$'+str(self.name.split(' ')[0])+'$', fontsize=18)
        ax.set_ylabel(r'$'+str(self.name.split(' ')[1])+'$', fontsize=18)

        #ax.legend(prop={'size': 16})

        if kwargs.get('save', False):
            filenm=kwargs.get('filename', OUTPUT_FIGURES+'/indices/index_plot_'+self.name.replace('/','_').replace('-', '_').replace(' ', '_')+'.pdf')
            plt.savefig(filenm, dpi=100, bbox_inches='tight')

    

    def plot(self, **kwargs):
        """
        Plotting function for an index-index space
    
        """
        dict1={'data':[self.templates[self.xkey].tolist(), self.templates[self.ykey].tolist()], 
               'name':'Templates', \
                'color':'#0074D9', 
                'marker':'D', 'ms':25, 'cmap':'YlOrBr', \
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
        
        if 'ax' in kwargs:
            ax1= kwargs.get('ax', None)
        else:
            fig=plt.figure(figsize=kwargs.get('figsize', (8,8)))
            ax1=fig.add_subplot(111)

        #plot contaminants
        x=np.array(dict3['data'][0])
        y=np.array(dict3['data'][1])
        
        small_df=pd.DataFrame([x, y]).applymap(np.float).transpose()
        small_df.columns=['x', 'y']
        small_df=small_df.replace([np.inf, -np.inf], np.nan)
        small_df=small_df.dropna(how='any')
        
        #
        #print (small_df)
        #only show things in range of the plot
        #small_df=small_df[(small_df.x.between(xlims[0],xlims[1]))& (small_df.y.between(ylims[0], ylims[1] ))]
        
        xd=small_df.x.as_matrix()  #rawr xd :) lol
        yd=small_df.y.as_matrix()
        
        #hist2d=ax1.hist2d(x=xd, y=yd, cmap=kwargs.get('cmap', MYCOLORMAP), bins=10, alpha=1.0)
        #cbar = fig.colorbar(hist2d[3],  orientation='horizontal')
        ax1.scatter(xd, yd,  marker='o',  facecolors='none',  edgecolors='#AAAAAA', label='Contaminants')

        #plot templates

        templates_data =[d['data'] for d in datadicts if d['name']=='Templates']
        templates_data=(np.array([*templates_data]))[0]
        #print (templates_data.shape)
        if templates_data != []:
            xmean=np.mean(templates_data[0][:,0])
            xstd=np.std(templates_data[0][:,0])
            ymean=np.mean(templates_data[1][:,0])
            ystd=np.std(templates_data[1][:,0])
            xmin=xmean-3.0*xstd
            ymin=ymean-3.0*ystd
            if xmin<0.0:
                xmin=0.0
            if ymin<0.0:
                ymin=0.0
            xlims=[xmin, xmean+3.0*xstd]
            ylims=[ymin, ymean+3.0*ystd]
            
        else:
            xlims=self.shapes[-1].xrange
            ylims= self.shapes[-1].yrange
    
            
        for d in datadicts:
            ax1.scatter(d['data'][0],d['data'][1],  facecolors=d['color'], marker='.',
                        s=d['ms'], cmap=d['cmap'], label=d['name'], alpha=d['alpha'], edgecolors=d['edgecolor'], 
                        linewidths=d['linewidths'])
                        
    
        #ax1.hist2d(x=xd, y=yd, cmap='Reds')
        df_=pd.DataFrame()
        df_['x']=d['data'][0]
        df_['y']=d['data'][1]
        #print (rmv)
        for s in self.shapes:
                if s.shape_name in kwargs.get('highlight', [None]): 
                    s.alpha=.5
                    s.plot( ax=ax1, only_shape=True, highlight=False, alpha=1.)
                else: pass
            #s.color='none'
            #print (""" itsssssssss """,  kwargs.get('highlight', None))
            #print ('name', s.shape_name, 'color', s.color, 'linewidth', s.linewidth)
            
        ax1.set_xlabel('$'+str(self.name.split(' ')[0])+'$', fontsize=18)
        ax1.set_ylabel('$'+str(self.name.split(' ')[1])+'$', fontsize=18)

        #plot manjavacas data
        #FF851
        mdf=Annotator.reformat_table(mjdf)
        sdf=Annotator.reformat_table(scndf)

        ax1.scatter(mdf[self.xkey], mdf[self.ykey],  marker='P', facecolors='none', edgecolors='#FF851B', label='Manjavacas')
        ax1.scatter(sdf[self.xkey], sdf[self.ykey],  marker='^', facecolors='none', edgecolors='#B10DC9', label='Schneider')

        
        #if kwargs.get('log_scale', False):
        #    plt.xscale('log')
        #    plt.yscale('log')
        #if np.std([s.xrange for s in self.shapes])>20.0*np.nanmedian([s.xrange for s  in self.shapes]):
        #    plt.xscale('log')
        #if np.std([s.yrange for s in  self.shapes])>20.0*np.nanmedian([s.yrange for s in self.shapes]):
        #    plt.yscale('log')
            
        filename=kwargs.get('filename', 'none')
        #set limits of the plts from templates 
        ax1.set_xlim(kwargs.get('xlim',  xlims))
        ax1.set_ylim(kwargs.get('ylim', ylims))

        #indices that use the continuum have ranges that are too high, logscale this?

        ax1.legend(prop={'size': 16})
        filenm=kwargs.get('filename', 
        OUTPUT_FIGURES+'/indices/index_plot_'+self.name.replace('/','_').replace('-', '_').replace(' ', '_')+'.pdf')
        if kwargs.get('save', True):
            plt.savefig(filenm, dpi=100, bbox_inches='tight')
            
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
    all_spex=datasets['spex']

    all_spex['Spts']=all_spex.spt.apply(splat.typeToNum).apply(float)
    all_spex['Names']=all_spex.grism_id

    tpl_ids=all_spex[all_spex['metallicity_cls'] !='d/sd']
    #templates['data_type']= 'templates'
    sd_ids=all_spex[all_spex['metallicity_cls'] =='d/sd']
    #subdwarfs['data_type']= 'subdwarf

    print (tpl_ids.shape, sd_ids.shape)
    
    conts=kwargs.get('conts', COMBINED_PHOTO_SPECTRO_DATA)
    #print(conts)
    conts=Annotator.reformat_table(conts.rename(columns={'grism_id': 'Names'}))
    keys=INDEX_NAMES
    index_spaces=[]
    for x_key, y_key in  tqdm(list(combinations(keys,2))):
        idspace=IndexSpace(xkey=x_key, ykey=y_key)
        #print (idspace.name)
        #print (idspace.xkey, idspace.ykey)
        #pass subdwarfs first 
        idspace.subdwarfs=sd_ids
        idspace.templates=tpl_ids[[x_key, y_key, 'Names', 'Spts']]
        #idspace._spex_sample=tpl_ids
        #annotated_df=Annotator.group_by_spt(tpl_ids)
        #idspace._calc_completeness(annotated_df)
        #print (len(idspace.templates[idspace.templates['spt_range']=='L0-L5']))
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
    crts=crts_from_file().values()
    conts=pd.DataFrame([ x.contamination for x in crts])
    compls=pd.DataFrame([ x.completeness for x in crts])
    conts['index-space']=[x.name for x in crts]
    compls['index-space']=[x.name for x in crts]
    compls.sort_index()
    conts.sort_index()
    
    conts.index=['Idx'+str(i) for i in range(0, len(conts['index-space']))]
    new_conts=conts.sort_values(by=list(conts.columns), ascending=False).drop(labels='index-space', axis=1)
    compls.index=['Idx'+str(i) for i in range(0, len(compls['index-space']))]
    new_compls=compls.sort_values(by=list(compls.columns), ascending=False).drop(labels='index-space', axis=1)
    
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=kwargs.get('figsize',(10, 6)), sharex=True, sharey=True)
    seaborn.heatmap(new_conts, cmap=cmap, ax=ax1)
    seaborn.heatmap(new_compls,  ax=ax2, cmap=cmap)
    
    ax2.set_title('Completeness', fontsize=16)
    ax1.set_title('Contamination', fontsize=16)
    
    fig.savefig( OUTPUT_FIGURES+'/completeness_contamination.pdf', bbox_inches='tight')
    
    return 
		
def pick_by_contamination(input_dict):
	"""
	Given a contamination cut and
	input dictionary should 
	"""


if __name__ =="__main__":
    crtFromfile()
   
