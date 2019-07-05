#dumping code
def galactic_density_juric(rc,zc):
    
# constants
    r0 = 8000.
    z0 = 25.
    l1 = 2600
    h1 = 300
    ftd = 0.12
    l2 = 3600
    h2 = 900.
    fh = 0.0051 # relative number of halo to thin disk star counts
    qh = 0.64 # halo axial ratio
    nh = 2.77 # halo power law index
    
    r = np.array(rc)+r0
    z = np.array(rc)+z0

    rhod0 = 1./(1.+ftd+fh)

# compute number densities of different components
    rhod = np.exp(-1.*(r-r0)/l1)*np.exp(-1.*np.absolute(z)/h1)
    rhotd = ftd*np.exp(-1.*(r-r0)/l2)*np.exp(-1.*np.absolute(z)/h2)
    rhoh = fh*rhod0*(((r0/(r**2+(z/qh)**2)**0.5))**nh)

# compensate for fact that we measure local density at the sun's position
   
    rhod = rhod*np.exp(z0/h1)
    rhotd = rhotd*np.exp(z0/h2) 
     
    return rhod+rhotd+rhoh

 def  _volume_correction(r, z):
    f = lambda r, z: galactic_density_juric(r, z)
    d=(r**2+z**2)**0.5
    return abs(d**3*integrate.dblquad(f, 0, r, lambda x: 0, lambda x: z)[0])

def volume_correction(r, z):
    if isinstance(r, (float, int)):
        return _volume_correction(r, z)
    if isinstance(r, Iterable):
        return np.array(list(itertools.starmap(_volume_correction,  map(list, zip(*[r, z])))))

class Pointing(object):
    
    def __init__(self, **kwargs):
        #only a function of coordinate
        self.coord=kwargs.get('coord', None)
        
        
    def rz(self, dist):
        """
        returns r and z given a distance
        """
        newcoord=SkyCoord(ra=self.coord.ra, dec=self.coord.dec, distance=dist*u.pc)
        r=(newcoord.cartesian.x**2+newcoord.cartesian.y**2)**0.5
        z=newcoord.cartesian.z
        return r, z
    
    def cdf(self, dmax, dnorm=10000):
        """
        Returns a CDF distribution of distances normalized at 10kpc from the sun
        input: 
              dmax= max distance to simulate up to
              dnorm=distance where CDF=1, normalization distance
        output: distance steps and asscoiated cdf values
        """
        rmax, zmax=self.rz(dnorm)
        norm=volume_correction(rmax.value, zmax.value)
    
        r, z=self.rz(dmax)
        corr=volume_correction(r.value, z.value)
    
        return corr/norm
    
    def random_draw(self, nsample=10, dnorm=10000):
        """
        random number from the CDF
        optional input: nsample, the number of points to return 
        """
        dvals=np.logspace(1.0, dnorm, 10)
        cdf_vals=self.cdf(dvals, dnorm=dnorm)
        
        x = np.random.rand( nsample)
        idx = [bisect.bisect(cdf_vals, i) for i in x]
        return dvals[idx]


class Pointing(object):
    def __init__(self, **kwargs):
        #only a function of coordinate
        self.coord=kwargs.get('coord', None)
    
    def cdf(self, dmax, dnorm=10000):
        """
        Returns a CDF distribution of distances normalized at 10kpc from the sun
        input: 
              dmax= max distance to simulate up to
              dnorm=distance where CDF=1, normalization distance
        output: distance steps and asscoiated cdf values
        """
        norm=dnorm**3*spsim.volumeCorrection(self.coord, 10**4, dmin=1.0, model='juric', center='sun', nsamp=10, unit=u.pc)
        dds=np.linspace(0, dmax, 1000)
        return dds, [x**3*spsim.volumeCorrection(self.coord, x, dmin=1.0, model='juric', center='sun', nsamp=10, unit=u.pc)/norm for x in dds]
    
    def random_draw(self, nsample=10, dnorm=10000):
        """
        random number from the CDF
        optional input: nsample, the number of points to return 
        """
        dvals, cdf_vals=self.cdf(dnorm, dnorm=dnorm)
        print (len(dvals))
        x = np.random.rand( nsample)
        idx = [bisect.bisect(cdf_vals, i) for i in x]
        return dvals[idx]