import numpy as np
from SALT2022_LineProfile import Line_Profile

# SALT parameters
alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0.0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0

# Turn Aperture and Occultation effect on or off (True or False)
OCCULTATION = True
APERTURE = True

# refence wavelength
lam_ref = 1193.28

# Observed velocity range centered on lam_ref
v_obs = np.linspace(-2000,2000,1000)

# Background to be scattered through SALT (this example assumes a flat continuum)                                                        
background = np.ones_like(v_obs)

# Outflow parameters
flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}

# Line Profile parameters
# abs_waves --> array/list of resonant absorption wavelengths, ordered from shortest to longest
# abs_osc_strs --> array/list of oscillator strengths :math:`f_{lu}` matching abs_waves in order and number
# em_waves --> array/list of resonant absorption wavelengths corresponding to each emission wavelength (includes resonance and fluorescence)
# em_osc_strs --> same as em_waves, but contains the associated oscillator strength for each absorption transition
# res --> array/list declares which lines in em_waves are resonant (True or False)
# fluor --> array/list declares which lines in em_waves are flourescent (True or False)
# p_r --> probability for resonance (see Scarlata and Panagia 2015 for definition)
# p_f --> probability for fluorescence (see Scarlata and Panagia 2015 for definition)
# final_waves --> determines all possible wavelengths for emission
# line_num --> list/array location corresponds to total # of absorption lines, number corresponds to number of emission lines resulting from the corresonding absorption
# v_obs --> list/array observed velocity values used in line profile, must contain enough room to capture all absorption/emission
# lam_ref --> reference wavelength for observed velocity range (i.e., location of zero observed velocity)
# APERTURE --> True or False determines if aperture is on or off
# OCCULTATION --> True or False determines if photons emitted from behind the source are blocked or not

profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

#Line_Profile --> output spectrum or line profile
spectrum  = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)


from matplotlib import pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(7, 5))

ax.plot(v_obs,spectrum,'r',linewidth = 2.0)

ax.set_xlabel('Velocity '+r'$[\rm{km} \ \rm{s}^{-1}]$',fontsize =20)
ax.set_ylabel(r'$F/F_0$',fontsize =20)
plt.grid()
plt.tight_layout()
plt.show()
