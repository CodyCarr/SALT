Fitting SALT to Data
====================

As a semi-analytical model, SALT is designed to compute spectral lines quickly.  When paired with a Monte Carlo sampler, one can constrain the SALT parameter space efficently to determine the properties of galactic outflows and the associated uncertainties.  In this tutorial, we fit SALT to the spectrum of a real galaxy using the Python based Monte Carlo ensemble sampler, emcee (Foreman-Mackey, Hogg, Lang & Goodman 2012).  The goal is to sufficiently sample a likelihood function to quantify the best parameter fit and associated uncertainties.  For this tutorial, we assume a Gaussian likelihood function.  The fitting code is described below.   

Fitting Code
************
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
       profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_range,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

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
       if 0<alpha<np.pi/2.0 and 0<psi<np.pi/2.0 and 0.5<gamma<2.0 and -2<tau<3 and 2.0<v_0<150.0 and 200.0<v_w<2500.0 and 0<f_c<1 and -2.0<k<2.0 and 0.5<delta<8.0 and 0<v_ap<2500:
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
       np.savetxt('chain_pars.txt',sampler.chain.reshape((-1, ndim)))
       np.savetxt('max_likelihood_pars.txt',sampler.get_log_prob().reshape((-1, nwalkers)))


    data = np.loadtxt('../../1244+0216.txt')
    v_obs = data[:0]
    flux = data[:1]
    error = data[:2]
    
    lam_ref = 1193.28
    v_range = np.linspace(int(v_obs[0])-1.0,int(v_obs[-1])+1,1500)

    res = 25.0/(v_range[1]-v_range[0])

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
    p0[:,9] = p0[:,9] * 600.0+200.0

    if __name__ == "__main__":
       main()

Results
*******

Here we analyize the results of the model fitting.  

.. code-block:: python

    chain = np.genfromtxt('/Users/codycarr/Desktop/Xshooter/xshooter/spectra2/'+names[s]+'/chain_pars.txt')
    ndim, nwalkers, steps = 10, 50, 3000
    chain = np.reshape(chain,(nwalkers,steps,ndim))

    likelihood = np.genfromtxt('/Users/codycarr/Desktop/Xshooter/xshooter/spectra2/'+names[s]+'/max_likelihood_pars.txt')
    likelihood = likelihood.ravel()

    best_fit_index = np.where(likelihood == max(likelihood))[0][0]

    alpha_chain = chain[:,:,0]
    psi_chain = chain[:,:,1]
    gamma_chain = chain[:,:,2]
    tau_chain = 10.0**chain[:,:,3]
    v_0_chain = chain[:,:,4]
    v_w_chain = chain[:,:,5]
    f_c_chain = chain[:,:,6]
    k_chain = 10.0**chain[:,:,7]
    delta_chain = chain[:,:,8]+chain[:,:,2]+2.0
    v_ap_chain = chain[:,:,9]
    
    alpha_array = np.array(alpha_chain.ravel())
    psi_array = np.array(psi_chain.ravel())
    gamma_array = np.array(gamma_chain.ravel())
    tau_array = np.array(tau_chain.ravel())
    v_0_array = np.array(v_0_chain.ravel())
    v_w_array = np.array(v_w_chain.ravel())
    f_c_array = np.array(f_c_chain.ravel())
    k_array = np.array(k_chain.ravel())
    delta_array = np.array(delta_chain.ravel())
    v_ap_array = np.array(v_ap_chain.ravel())

    best_fit = [alpha_array[best_fit_index],psi_array[best_fit_index],gamma_array[best_fit_index],tau_array[best_fit_index],v_0_array[best_fit_index],v_w_array[best_fit_index],f_c_array[best_fit_index],k_array[best_fit_index],delta_array[best_fit_index],v_ap_array[best_fit_index]]

    # best fit plot

    alpha = best_fit[0]
    psi = best_fit[1]
    gamma = best_fit[2]
    tau = best_fit[3]
    v_0 = best_fit[4]
    v_w = best_fit[5]
    f_c = best_fit[6]
    k = best_fit[7]
    delta = best_fit[8]
    v_ap = best_fit[9]

    lam_ref = 1193.28
    v_obs = np.linspace(-2000,2000,1000)
    background = np.ones_like(v_obs)
    OCCULTATION = True
    APERTURE = True
    flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
    profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}
    spectrum  = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1,1, figsize=(7, 5))
    ax.plot(v_obs,spectrum,'r',linewidth = 2.0)
    ax.set_xlabel('Velocity '+r'$[\rm km \ s^{-1}]$',fontsize =20)
    ax.set_ylabel(r'$F/F_0$',fontsize =20)
    plt.grid()
    plt.tight_layout()
    plt.show()

    
    # steps and walkers plots
    fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(9,1,figsize=(10,8.5))

    ax1.plot((180/np.pi)*gdchain[:,:,0].T,alpha=0.25)
    ax2.plot((180/np.pi)*gdchain[:,:,1].T,alpha=0.25)
    ax3.plot(gdchain[:,:,2].T,alpha=0.25)
    ax4.plot(gdchain[:,:,3].T,alpha=0.25)
    ax5.plot(gdchain[:,:,4].T,alpha=0.25)
    ax6.plot(gdchain[:,:,5].T,alpha=0.25)
    ax7.plot(gdchain[:,:,6].T,alpha=0.25)
    ax8.plot(gdchain[:,:,7].T,alpha=0.25)
    ax9.plot(gdchain[:,:,8].T,alpha=0.25)

    ax1.set_ylabel(r'$\alpha$',fontsize = 30, labelpad = 18)
    ax1.set_yticks([15,45,75])
    ax1.set_yticklabels([15,45,75])
    ax1.set_xticklabels([])

    ax2.set_ylabel(r'$\psi$',fontsize = 30, labelpad = 18)
    ax2.set_yticks([15,45,75])
    ax2.set_yticklabels([15,45,75])
    ax2.set_xticklabels([])

    ax3.set_ylabel(r'$\gamma$',fontsize = 30, labelpad = 25)
    ax3.set_yticks([.5,1,1.5])
    ax3.set_yticklabels([.5,1,1.5])
    ax3.set_xticklabels([])

    ax4.set_ylabel(r'$\tau$',fontsize = 30, labelpad = 18)
    ax4.set_yticks([20,40,60,80])
    ax4.set_yticklabels([20,40,60,80])
    ax4.set_xticklabels([])

    ax5.set_ylabel(r'$v_0$',fontsize = 30, labelpad = 20)
    ax5.set_yticks([40,80,120])
    ax5.set_yticklabels([40,80,120])
    ax5.set_xticklabels([])

    ax6.set_ylabel(r'$v_{\infty}$',fontsize = 30)
    ax6.set_yticks([500,1000,1500,2000])
    ax6.set_yticklabels([500,1000,1500,2000])
    ax6.set_xticklabels([])

    ax7.set_ylabel(r'$f_c$',fontsize = 30, labelpad = 18)
    ax7.set_yticks([.2,.4,.6,.8])
    ax7.set_yticklabels([.2,.4,.6,.8])
    ax7.set_xticklabels([])

    ax8.set_ylabel(r'$\kappa$',fontsize = 30, labelpad = 18)
    ax8.set_yticks([1,2,3,4,5])#20,40,60,80])                                                                                                         
    ax8.set_yticklabels([1,2,3,4,5])
    ax8.set_ylim([0,5])
    
    ax9.set_ylabel(r'$\delta$',fontsize = 30, labelpad = 18)
    ax9.set_yticks([2,4,6,8])#20,40,60,80])                                                                                                          
    ax9.set_yticklabels([2,4,6,8])
    ax9.set_ylim([0,8])

    ax1.set_ylabel(r'$\alpha$',fontsize = 30)
    ax2.set_ylabel(r'$\psi$',fontsize = 30)
    ax3.set_ylabel(r'$\gamma$',fontsize = 30)
    ax4.set_ylabel(r'$\tau$',fontsize = 30)
    ax5.set_ylabel(r'$v_0$',fontsize = 30)
    ax6.set_ylabel(r'$v_{\infty}$',fontsize = 30)
    ax7.set_ylabel(r'$f_c$',fontsize = 30)
    ax8.set_ylabel(r'$\kappa$',fontsize = 30)
    ax9.set_ylabel(r'$\delta$',fontsize = 30)

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
       ax.set_xlabel('Number of Steps',fontsize = 30)
       fig = plt.gcf()
    plt.show()
    plt.close(fig)

    # pdfs 

    import triangle
    tmp = triangle.corner(sampler.flatchain, labels=['alpha','betax','betay','eps'],truths=[alpha_true, beta_x_true, beta_y_true, eps_true])
