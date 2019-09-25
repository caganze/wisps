
functions {
  real galactic_density(real z, real r, real h) {
    
    real r0 = 8000; /*  radial offset from galactic center to Sun */
    real z0=25 ; /* z offset for the sun
    real l = 2600 ; /* radial length scale of exponential thin disk */
    real fh = 0.0051; /* relative number of halo to thin disk star counts */
    real qh = 0.64;  /* halo axial ratio */
    real nh = 2.77; /* halo power law index */
    
    real rhod = 1 ./(1+fh);
    
    real  rr=r-r0;
    real  zz=z-z0;
    
    real disk_num=exp(-1*rr/h);
    real disk_den= square(cosh(zz/h));
    
    real disk=disk_num ./ disk_den;
    
    real  halo;
    
    halo=rhod*fh*(pow(r0/sqrt(square(r)+ square(z/qh)), nh)); 
     
    return disk+halo;
    }
    
  real probability_density_lpdf(real r, real z, real h)
  {
  
      real d=square(r) + square(z);
      real density =galactic_density(z, r, h);
      real result = d* density;
      return log(result);
  }
  
  
}



parameters {
  real z ;
  real r;
  real h;
 }


model {
  /* Impose some weak priors on the parameters */
  h  ~ uniform(0, 10000);
  z ~ probability_density_lpdf(r, h);
  r ~ probability_density_lpdf(z, h);
  
}
