
from .initialize import *
from scipy import stats
from ..utils.tools import drop_nan, splat_teff_to_spt,kernel_density
from tqdm import tqdm
import splat.simulate as spsim
import splat.evolve as spev
import splat.empirical as spe
import wisps
#import pymc3 as pm
from scipy.interpolate import griddata
#import theano.tensor as tt
#from theano.compile.ops import as_op
import astropy.units as u
import numba

BINARY_TABLE=pd.read_pickle(wisps.OUTPUT_FILES+'/binary_lookup_table.pkl.gz')
BINARY_TABLE_SYS=(BINARY_TABLE['sys']).values
BINARY_TABLE_PRIM=(BINARY_TABLE['prim']).values
BINARY_TABLE_SEC=(BINARY_TABLE['sec']).values

def log_mass_function(m, alpha):
    """
    Power law mass function
    """
    return np.log(m**-alpha)

def log_mass_ratio(q, gamma):
    """
    Power law mass ratio
    m1 is primary mass
    m2 is secondary mass """
    return np.log(q**gamma)

def total_likelihood(m1, q, alpha, gamma):
    return log_mass_function(m1, alpha)+log_mass_ratio(q, gamma)


def simulate_binary(nstuff, mass_range, age_range):
    """
    Simulate a distribution of binaries from simple assumptions
    This is much faster than splat
    """
    gamma=4
    with pm.Model() as model:
        alpha=0.6
        prim=pm.Uniform('m1', lower=mass_range[0], upper=mass_range[1]) #primaries
        q=pm.Uniform('q', lower=.1, upper=1.)

        sec=pm.Deterministic('m2', prim*q)
        age=pm.Uniform('t', lower=age_range[0], upper=age_range[-1]) #system age
        like = pm.Potential('likelihood', total_likelihood(prim, q, alpha, gamma))
        trace = pm.sample(draws=nstuff,  cores=4,  tune=int(nstuff/20),  init='advi')

    return [trace.m1, trace.m2, trace.t]

def get_system_type(pr, sc):
    """
    use the lookup table to get a spectral type for the binary
    using a linear interpolation to avoid nans
    """
    #where secondary are nans set to primaries
    sc[np.isnan(sc)]=pr[np.isnan(sc)]
    interpoints=np.array([BINARY_TABLE_PRIM, BINARY_TABLE_SEC ]).T
    comb=griddata(interpoints, BINARY_TABLE_SYS , (pr, sc), method='linear')
    return comb


def evolutionary_model_interpolator(mass, age, model):
    """
    My own evolutionary model interpolator, 

    Hoping to make it faster than splat because I'm impatient 

    input: mass, age
    model: model name
    """

    model_filename=EVOL_MODELS_FOLDER+'//'+model.lower()+'.csv'
    evolutiomodel=pd.read_csv( model_filename)

    #use the full cloud treatment for saumon models
    if model=='saumon2008':
         evolutiomodel=evolutiomodel[evolutiomodel.cloud=='hybrid']
 
    #make age, teff, mass logarithm scale
    valuest=np.log10(evolutiomodel.temperature.values)
    valueslogg=evolutiomodel.gravity.values
    valueslumn=evolutiomodel.luminosity.values

    valuesm=np.log10(evolutiomodel.mass.values)
    valuesag=np.log10(evolutiomodel.age.values)

    evolpoints=np.array([valuesm, valuesag ]).T

    teffs=griddata(evolpoints, valuest , (np.log10(mass), np.log10(age)), method='linear')
    lumn=griddata(evolpoints, valueslumn , (np.log10(mass), np.log10(age)), method='linear')


    return {'mass': mass*u.Msun, 'age': age*u.Gyr, 'temperature': 10**teffs*u.Kelvin, 
    'luminosity': lumn*u.Lsun}




def simulate_spts(**kwargs):
    """
    Simulate parameters from mass function,
    mass ratio distribution and age distribution
    """
    recompute=kwargs.get('recompute', False)
    model_name=kwargs.get('name','baraffe2003')

    #use hybrid models that predit the T dwarf bump for Saumon Models
    if model_name=='saumon2008':
        cloud='hybrid'
    else:
        cloud=False

    #automatically set maxima and minima to avoid having too many nans
    #mass age and age,  min, max
    #all masses should be 0.01
    acceptable_values={'baraffe2003': [0.01, 0.1, 0.01, 8.0],
    'marley2019': [0.01, 0.08, 0.001, 8.0], 'saumon2008':[0.01, 0.09, 0.003, 8.0], 
    'phillips2020':[0.01, 0.075, 0.001, 8.0 ]}
    
    if recompute:

        nsim = kwargs.get('nsample', 1e5)

        ranges=acceptable_values[model_name]
        
        # masses for singles [this can be done with pymc but nvm]
        m_singles = spsim.simulateMasses(nsim,range=[ranges[0], ranges[1]],distribution='power-law',alpha=0.6)
        #ages for singles
        ages_singles= spsim.simulateAges(nsim,range=[ranges[2], ranges[3]], distribution='uniform')

        #parameters for binaries
        #binrs=simulate_binary(int(nsim), [ranges[0], ranges[1]], [ranges[2], ranges[3]])
        qs=spsim.simulateMassRatios(nsim,distribution='power-law',q_range=[0.1,1.0],gamma=4)
        m_prims = spsim.simulateMasses(nsim,range=[ranges[0], ranges[1]],distribution='power-law',alpha=0.6)
        m_sec=m_prims*qs
        ages_bin= spsim.simulateAges(nsim,range=[ranges[2], ranges[3]], distribution='uniform')

        #single_evol=spev.modelParameters(mass=m_singles,age=ages_singles, set=model_name, cloud=cloud)
        single_evol=evolutionary_model_interpolator(m_singles, ages_singles, model_name)

        #primary_evol=spev.modelParameters(mass=binrs[0],age=binrs[-1], set=model_name, cloud=cloud)
        primary_evol=evolutionary_model_interpolator(m_prims,ages_bin, model_name)

        #secondary_evol=spev.modelParameters(mass=binrs[1],age=binrs[-1], set=model_name, cloud=cloud)
        secondary_evol=evolutionary_model_interpolator(m_sec,ages_bin, model_name)
        #save luminosities

        #temperatures
        teffs_singl =single_evol['temperature'].value
        teffs_primar=primary_evol['temperature'].value
        teffs_second=secondary_evol['temperature'].value

        #spectraltypes
        spts_singl =splat_teff_to_spt(teffs_singl)

        #the singles will be fine, remove nans from systems 
        spt_primar=splat_teff_to_spt(teffs_primar)
        spt_second=splat_teff_to_spt(teffs_second)

        #remove nans 
        print ('MAX AGES', np.nanmax(ages_singles))
        #print ('MAX AGES', np.nanmax())

        xy=np.vstack([np.round(np.array(spt_primar), decimals=0), np.round(np.array(spt_second), decimals=0)]).T

        spt_binr=get_system_type(xy[:,0], xy[:,1])

   
        values={ 'sing_evol': single_evol, 'sing_spt':spts_singl,
        		 'prim_evol': primary_evol, 'prim_spt':spt_primar,
        		 'sec_evol': secondary_evol, 'sec_spt': spt_second,
        		'binary_spt': spt_binr }

        import pickle
        with open(wisps.OUTPUT_FILES+'/mass_age_spcts_with_bin{}.pkl'.format(model_name), 'wb') as file:
           pickle.dump(values,file)
    else:
        values=pd.read_pickle(wisps.OUTPUT_FILES+'/mass_age_spcts_with_bin{}.pkl'.format(model_name))


    return values

def make_systems(**kwargs):
    """
    choose a random sets of primaries and secondaries 
    and a sample of single systems based off a preccomputed-evolutionary model grid 
    and an unresolved binary fraction

    """
    #recompute for different evolutionary models
    model=kwargs.get('model_name', 'baraffe2003')
    binary_fraction=kwargs.get('bfraction', 0.2)

    model_vals=simulate_spts(name=model, **kwargs)


    #nbin= int(len(model_vals['sing_spt'])*binary_fraction) #number of binaries
    ndraw= int(len(model_vals['sing_spt'])/(1-binary_fraction))-int(len(model_vals['sing_spt']))


    nans=np.isnan(model_vals['binary_spt'])
    
    choices={'spt': np.random.choice(model_vals['binary_spt'][~nans], ndraw),
            'teff': np.random.choice(model_vals['prim_evol']['temperature'].value[~nans], ndraw), 
            'age': np.random.choice(model_vals['prim_evol']['age'].value[~nans],ndraw)}


    vs={'system_spts': np.concatenate([model_vals['sing_spt'], choices['spt']]), 
            'system_teff':  np.concatenate([(model_vals['sing_evol']['temperature']).value, choices['teff']]),
            'system_age':  np.concatenate([(model_vals['sing_evol']['age']).value,  choices['age']])}

    return vs