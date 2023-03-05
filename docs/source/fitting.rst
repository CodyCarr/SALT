Fitting SALT to Data
====================

Fitting the SALT model to data with the Python software, emcee

emcee Monte Carlo
*****************
.. code-block:: python

    import numpy as np
    import scipy.ndimage.filters as g
    from scipy import interpolate
    import emcee

    from SALT2022_LineProfile import Line_Profile

    def rebin(x,y,xnew):
       f = interpolate.interp1d(x,y)
       ynew = f(xnew)
       return ynew

    def lnlike(pars, y, yerr):

       alpha,psi,gamma, tau, v_0, v_w, f_c, k, delta, v_ap = pars
       
       tau = 10.0**tau
       k = 10.0**k
       delta = gamma+2.0+delta

       background = np.ones_like(v_range)
       
       OCCULTATION = True
       APERTURE = True

       flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
       profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

       model = Line_Profile(v_range,lam_ref,background,flow_parameters,profile_parameters)
       model = np.array(g.gaussian_filter1d(model,res))
       model = rebin(v_range,model,v_obs)

       result = 0
       for i in range(len(list(y))):
          sigma = 1.0/yerr[i]**2.0
          result += (((y[i]-model[i])**2.0*sigma)-np.log(sigma))
       result = -.5*result
       return result

    def lnprior(pars):
       alpha,psi,gamma,tau,v_0,v_w,f_c,k,delta,v_ap = pars
       if 0<alpha<np.pi/2.0 and 0<psi<np.pi/2.0 and 0.5<gamma<2.0 and -2<tau<3 and 2.0<v_0<150.0 and 200.0<v_w<2500.0 and 0<f_c<1 and -2.0<k<2.0  and 0.5<delta<8.0 and 200.0<a<2500.0:
          return 0
       return -np.Inf

    def lnprob(pars,y,yerr):
       lnp = lnprior(pars)
       if not np.isfinite(lnp):
          return -np.Inf
       return lnp + lnlike(pars,y,yerr)
       
    def main():
       sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(np.array(flux), np.array(error)), pool=Pool(max_workers = 25))
       start_time = time.time()
       sampler.run_mcmc(p0, steps);
       print("--- %s seconds ---" % (time.time() - start_time))
       np.savetxt('~/chain_pars.txt',sampler.chain.reshape((-1, ndim)))
       np.savetxt('~/max_likelihood_pars.txt',sampler.get_log_prob().reshape((-1, nwalkers)))


    v_obs,flux,error = np.loadtxt('~/../../1244+0216.txt')

    lam_ref = 1193.28
    v_range = np.linspace(int(v_obs[0])-1.0,int(v_obs[-1])+1,1500)

    res = 55.517/(v_range[1]-v_range[0])

    ndim, nwalkers, steps = 10, 50, 3000
    p0 = np.random.rand(nwalkers,ndim)

    p0[:,0] = p0[:,0] * np.pi/2.0
    p0[:,1] = p0[:,1] * np.pi/2.0
    p0[:,2] = p0[:,2] * 1.5+.5
    p0[:,3] = p0[:,3] * 5.0-2.0
    p0[:,4] = p0[:,4] * 98. +2.
    p0[:,5] = p0[:,5] * 600.0+200.0
    p0[:,6] = p0[:,6]
    p0[:,7] = p0[:,7] * 4.0-2.0
    p0[:,8] = p0[:,8] * 3.0-1.5
    p0[:,9] = p0[:,9]*600.0+200.0

    if __name__ == "__main__":
       main()
