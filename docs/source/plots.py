import numpy as np
import scipy.ndimage.filters as g
from scipy import interpolate
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '/Users/codycarr/Desktop/repository/SALT/python/')
from SALT2022_LineProfile import Line_Profile

def rebin(x,y,xnew):
   f = interpolate.interp1d(x,y)
   ynew = f(xnew)
   return ynew

# get data
data = np.loadtxt('../../1244+0216.txt')
v_obs = data[:,0]
flux = data[:,1]
error = data[:,2]
eu = flux+error
ed = flux-error

# get chains
chain = np.genfromtxt('chain_pars.txt')
ndim, nwalkers, steps = 10, 50, 3000
chain = np.reshape(chain,(nwalkers,steps,ndim))

# collect chains
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

# convert chains to arrays
alpha_arr = np.array(alpha_chain.ravel())
psi_arr = np.array(psi_chain.ravel())
gamma_arr = np.array(gamma_chain.ravel())
tau_arr = np.array(tau_chain.ravel())
v_0_arr = np.array(v_0_chain.ravel())
v_w_arr = np.array(v_w_chain.ravel())
f_c_arr = np.array(f_c_chain.ravel())
k_arr = np.array(k_chain.ravel())
delta_arr = np.array(delta_chain.ravel())
v_ap_arr = np.array(v_ap_chain.ravel())

# find best fit from likelihood samples
likelihood = np.genfromtxt('max_likelihood_pars.txt').ravel()
bf_ind = np.where(likelihood == max(likelihood))[0][0]
best_fit = [alpha_arr[bf_ind],psi_arr[bf_ind],gamma_arr[bf_ind],tau_arr[bf_ind],v_0_arr[bf_ind],v_w_arr[bf_ind],f_c_arr[bf_ind],k_arr[bf_ind],delta_arr[bf_ind],v_ap_arr[bf_ind]]

# best fit SALT parameters
alpha,psi,gamma,tau,v_0,v_w,f_c,k,delta,v_ap = best_fit

# compute SALT
lam_ref = 1193.28
v_range = np.linspace(int(v_obs[0])-1.0,int(v_obs[-1])+1,1500)
background = np.ones_like(v_range)
OCCULTATION = True
APERTURE = True
flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_range,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}
spectrum  = Line_Profile(v_range,lam_ref,background,flow_parameters,profile_parameters)

# smooth and rebin data
res = 30.0/(v_range[1]-v_range[0])
spectrum = np.array(g.gaussian_filter1d(spectrum,res))
spectrum = rebin(v_range,spectrum,v_obs)

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(7, 5))
ax.fill_between(v_obs, eu, ed,alpha = .5,color = 'grey')
ax.step(v_obs,flux,'k',linewidth = 2,label='observed')
ax.plot(v_obs,spectrum,'r',linewidth = 2.0,label='SALT')
ax.set_xlabel('Velocity '+r'$[\rm km \ s^{-1}]$',fontsize =20)
ax.set_ylabel(r'$F/F_0$',fontsize =20)
ax.legend(loc='upper left',fontsize = 20,edgecolor = 'white',facecolor = 'white',framealpha=0.8)
plt.grid()
plt.tight_layout()
plt.show()
