import numpy as np
import scipy.ndimage.filters as g
from matplotlib import pyplot as plt
from scipy import interpolate
import sys
sys.path.insert(0, '/Users/codycarr/Desktop/repository/SALT/python/')
from SALT2022_LineProfile import Line_Profile

def rebin(x,y,xnew):
   f = interpolate.interp1d(x,y)
   ynew = f(xnew)
   return ynew

chain = np.genfromtxt('chain_pars.txt')
ndim, nwalkers, steps = 10, 50, 3000
chain = np.reshape(chain,(nwalkers,steps,ndim))

likelihood = np.genfromtxt('max_likelihood_pars.txt')
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

data = np.loadtxt('../../1244+0216.txt')
v_obs = data[:,0]
flux = data[:,1]
error = data[:,2]

lam_ref = 1193.28
v_range = np.linspace(int(v_obs[0])-1.0,int(v_obs[-1])+1,1500)
res = 25.0/(v_range[1]-v_range[0])
background = np.ones_like(v_range)
OCCULTATION = True
APERTURE = True
flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
print(flow_parameters)
profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_range,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

spectrum = Line_Profile(v_range,lam_ref,background,flow_parameters,profile_parameters)
spectrum = np.array(g.gaussian_filter1d(spectrum,res))
spectrum = rebin(v_range,spectrum,v_obs)

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(7, 5))
ax.step(v_obs,flux,'k',linewidth = 1.0)
ax.plot(v_obs,spectrum,'r',linewidth = 2.0)
ax.set_xlabel('Velocity '+r'$[\rm km \ s^{-1}]$',fontsize =20)
ax.set_ylabel(r'$F/F_0$',fontsize =20)
plt.grid()
plt.tight_layout()
plt.show()
