Fitting SALT to Data
====================

Fitting the SALT model to data with the Python software, emcee

emcee Monte Carlo
*****************
.. code-block:: python

    import numpy as np
    import math as m
    import scipy as s
    import matplotlib.pyplot as plt
    import scipy.ndimage.filters as g
    from scipy import interpolate
    import emcee

    def lnlike(pars, y, yerr):

       alpha,psi,gamma, tau, v_0, v_w, f_c, k, delta, a, aa, c, cc = pars
       v_ap = v_w
       a = a
       aa = aa
       tau = 10.0**tau
       k = 10.0**k
       b = 0
       bb = shift
       Gauss = [a,b,c,aa,bb,cc]
       delta = gamma+2.0+delta

       parameters1 = [[alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,delta,APERTURE],[alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,delta,APERTURE]]
       Normalized_Flux = a*np.exp(-(v_range-b)**2.0/(2.0*c**2.0))+1.0+aa*np.exp(-(v_range-bb)**2.0/(2.0*cc**2.0))

       #computes list of absorption files for each resonant line  
       with Pool(max_workers=2) as inner_pool:
          Absorption_Profiles = list(inner_pool.map(partial(Function_CallABS,parameters1),range(2)))

       Normalized_Flux_Lists = makeAbsorptionPROFILE(abs_wavelengths,abs_oscillator_strengths,lambda_ref,v_range,Normalized_Flux,parameters1,Absorption_Profiles)

       parameters2 = [[alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,k,delta,APERTURE,True,False,OCCULTATION,1.0,0.0],[alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,k,delta,APERTURE,True,False,OCCULTATION,1.0,0.0]]

       New_Flux_Lists = [Normalized_Flux_Lists[0],Normalized_Flux_Lists[1]]

       with Pool(max_workers=4) as inner_pool:
          Emission_Profiles = list(inner_pool.map(partial(Function_CallEM,parameters2,New_Flux_Lists),range(2)))
       Emission = makeEmissionPROFILE(profile_wavelengths,em_oscillator_strengths,lambda_ref,v_range,Normalized_Flux,parameters2,Emission_Profiles)
       Absorption = Normalized_Flux_Lists[-1]
       model = Absorption+Emission
       model = np.array(g.gaussian_filter1d(model,res))
       model = rebin(v_range,model,v_obs)

       result = 0
       for i in range(len(list(y))):
          sigma = 1.0/yerr[i]**2.0
          result += (((y[i]-model[i])**2.0*sigma)-np.log(sigma))
       result = -.5*result
       return result

    def lnprior(pars):
       alpha,psi,gamma,tau,v_0,v_w,f_c,k,delta,a,aa,c,cc = pars
       if 0<alpha<np.pi/2.0 and 0<psi<np.pi/2.0 and 0.5<gamma<2.0 and -2<tau<3 and 2.0<v_0<150.0 and 200.0<v_w<2500.0 and 0<f_c<1 and -2.0<k<2.0  and 0.5<delta<8.0 and 0.0<a<4.0 and 0.0<aa<4.0 and 0.0<c<300.0 and 0.0<cc<300.0:
          return 0
       return -np.Inf
