
from .initialize import *
from scipy import stats
from ..utils.tools import drop_nan, splat_teff_to_spt,kernel_density
from tqdm import tqdm
import splat.simulate as spsim
import splat.evolve as spev
import wisps
import pymc3 as pm
from scipy.interpolate import griddata
import theano.tensor as tt
from theano.compile.ops import as_op
import astropy.units as u

BINARY_TABLE=pd.read_pickle(wisps.OUTPUT_FILES+'/binary_lookup_table.pkl')

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

def total_likelihood(m1, m2, alpha, gamma):
    return log_mass_function(m1, alpha)+log_mass_function(m2, alpha)+log_mass_ratio(m1/m2, gamma)


def simulate_binary(nstuff):
    """
    Simulate a distribution of binaries from simple assumptions
    This is much faster than splat
    """
    gamma=4
    with pm.Model() as model:
        alpha=0.6
        prim=pm.Uniform('m1', lower=0.001, upper=.2) #primaries
        q=pm.Uniform('q', lower=0.1, upper=1.)
        sec=pm.Deterministic('m2', prim/q)
        age=pm.Uniform('t', lower=0.1, upper=10) #system age
        like = pm.DensityDist('likelihood', total_likelihood, observed={'m1': prim, 'm2': sec, 
	                                                                   'alpha': alpha, 'gamma': gamma})
        trace = pm.sample(draws=nstuff,  cores=4,  tune=int(nstuff/20), discard_tuned_samples=True, init='advi')

    return [trace.m1, trace.m2, trace.t]


@np.vectorize
def get_system_type(pr, sc):
    """
    use the lookup table to get a spectral type for the binary
    """
    return np.nanmean(BINARY_TABLE['sys'][(BINARY_TABLE.prim==np.round(pr)) &(BINARY_TABLE.sec==np.round(sc))])


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

    valuest=evolutiomodel.temperature.values
    valueslogg=evolutiomodel.gravity.values
    valuesrads=evolutiomodel.radius.values
    valueslumn=evolutiomodel.luminosity.values

    evolpoints=np.array([evolutiomodel.mass.values, evolutiomodel.age.values]).T

    teffs=griddata(evolpoints, valuest , (mass, age), method='linear')
    lumn=griddata(evolpoints, valueslumn , (mass, age), method='linear')

    return {'mass': mass*u.Msun, 'age': age*u.Gyr, 'temperature': teffs*u.Kelvin, 'luminosity': lumn*u.Lsun}




def simulate_spts(**kwargs):
    """
    add binaries
    """
    recompute=kwargs.get('recompute', False)
    model_name=kwargs.get('name','baraffe2003')

    #use hybrid models that predit the T dwarf bump for Saumon Models
    if model_name=='saumon2008':
        cloud='hybrid'
    else:
        cloud=False
    
    if recompute:

        nsim = kwargs.get('nsample', 1e5)
        
        # masses for singles [this can be done with pymc but nvm]
        m_singles = spsim.simulateMasses(nsim,range=[0.001,0.2],distribution='power-law',alpha=0.6)
        #ages for singles
        ages_singles= spsim.simulateAges(nsim,range=[0.1,10.], distribution='uniform')

        #parameters for binaries
        binrs=simulate_binary(int(nsim))
        #evol parameters
        #single_evol=spev.modelParameters(mass=m_singles,age=ages_singles, set=model_name, cloud=cloud)
        single_evol=evolutionary_model_interpolator(m_singles, ages_singles, model_name)


        #primary_evol=spev.modelParameters(mass=binrs[0],age=binrs[-1], set=model_name, cloud=cloud)
        primary_evol=evolutionary_model_interpolator(binrs[0],binrs[-1], model_name)


        #secondary_evol=spev.modelParameters(mass=binrs[1],age=binrs[-1], set=model_name, cloud=cloud)
        secondary_evol=evolutionary_model_interpolator(binrs[1],binrs[-1], model_name)
        #save luminosities


        #temperatures
        teffs_singl =single_evol['temperature'].value
        teffs_primar=primary_evol['temperature'].value
        teffs_second=secondary_evol['temperature'].value

        #spectratypes
        spts_singl = splat_teff_to_spt(teffs_singl)
        spt_primar=splat_teff_to_spt(teffs_primar)
        spt_second=splat_teff_to_spt(teffs_second)
        spt_binr=get_system_type(spt_primar, spt_second)

   
        values={ 'sing_evol': single_evol, 'sing_spt':spts_singl,
        		 'prim_evol': primary_evol, 'prim_spt':spt_primar,
        		 'sec_evol': secondary_evol, 'sec_spt': spt_second,
        		'binary_spt': spt_binr}

        import pickle
        with open(wisps.OUTPUT_FILES+'/mass_age_spcts_with_bin{}.pkl'.format(model_name), 'wb') as file:
           pickle.dump(values,file)
    else:
        values=pd.read_pickle(wisps.OUTPUT_FILES+'/mass_age_spcts_with_bin{}.pkl'.format(model_name))


    return values

def make_systems(**kwargs):
    """
    Purpose:who knows
    """

    #recompute for different evolutionary models
    model=kwargs.get('model_name', 'baraffe03')
    binary_fraction=kwargs.get('bfraction', 0.1)
    model_vals=simulate_spts(name=model)

    nbin= int(len(model_vals['sing_spt'])*binary_fraction) #number of binaries

    nan_idx=np.isnan(model_vals['binary_spt'])

    vs={'system_spts': np.concatenate([model_vals['sing_spt'], ((model_vals['binary_spt'])[~nan_idx])[:nbin]]),
     		'system_age':  np.concatenate([(model_vals['sing_evol']['age']).value, ((model_vals['prim_evol']['age']).value)[:nbin]]),
     		'system_mass': np.concatenate([model_vals['sing_evol']['mass'].value, 
                                     (model_vals['prim_evol']['mass'].value)[:nbin]+(model_vals['sec_evol']['mass'].value)[:nbin]])}

    return vs