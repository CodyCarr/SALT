import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from SALT2022_LineProfile import Line_Profile

#### Examples of line profiles made with SALT

#### SiII 1260

v_obs = np.linspace(-1500,2500,1000)
lam_ref = 1260.42

background = np.ones_like(v_obs)

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1260.42],'abs_osc_strs':[1.22], 'em_waves':[1260.42,1260.42],'em_osc_strs':[1.22,1.22],'res':[True,False],'fluor':[False,True],'p_r':[0.45811051693404636,0.45811051693404636],'p_f':[0.5418894830659536,0.5418894830659536],'final_waves':[1260.42,1265.02],'line_num':[2], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### SiII 1190,1193

v_obs = np.linspace(-2000,2000,1000)
lam_ref = 1193.28

shift = ((1190.42-1193.28)/(1193.28))*(2.99792458*10**5)
shift2 = ((1194.5-1193.28)/(1193.28))*(2.99792458*10**5)
shift3 = ((1197.39-1193.28)/(1193.28))*(2.99792458*10**5)
a,b,c = -1.0,shift,75
aa,bb,cc = -1.0,0,75

background = np.ones_like(v_obs)#a*np.exp(-(v_obs-b)**2.0/(2.0*c**2.0))+1.0+aa*np.exp(-(v_obs-bb)**2.0/(2.0*cc**2.0))-a*np.exp(-(v_obs-shift2)**2.0/(2.0*cc**2.0))-aa*np.exp(-(v_obs-shift3)**2.0/(2.0*cc**2.0))

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1190.42,1193.28],'abs_osc_strs':[0.277,.575], 'em_waves':[1190.42,1190.42,1193.28,1193.28],'em_osc_strs':[0.277,0.277,0.575,0.575],'res':[True,False,True,False],'fluor':[False,True,False,True],'p_r':[.1592,.1592,.6577,.6577],'p_f':[.8408,.8408,.3423,.3423],'final_waves':[1190.42,1194.5,1193.28,1197.39],'line_num':[2,2], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### SiIII 1206

v_obs = np.linspace(-1000,1000,1000)
lam_ref = 1206.5

background = np.ones_like(v_obs)

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1206.5],'abs_osc_strs':[1.67], 'em_waves':[1206.5],'em_osc_strs':[1.67],'res':[True],'fluor':[False],'p_r':[1.0],'p_f':[0.0],'final_waves':[1206.5],'line_num':[1], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### SiIV 1394,1403

v_obs = np.linspace(-1000,3000,1000)
lam_ref = 1393.76

background = np.ones_like(v_obs)

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1393.76,1402.77],'abs_osc_strs':[.513,.255], 'em_waves':[1393.76,1402.77],'em_osc_strs':[.513,.255],'res':[True,True],'fluor':[False,False],'p_r':[1.0,1.0],'p_f':[0.0,0.0],'final_waves':[1393.76,1402.77],'line_num':[1,1], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### CIV 1548.202,1550.772

v_obs = np.linspace(-750,1000,1000)
lam_ref = 1548.202

shift = ((1550.772-1548.202)/(1548.202))*(2.99792458*10**5)
a,b,c = 1.0,0.0,75
aa,bb,cc = 1.0,shift,75

background = a*np.exp(-(v_obs-b)**2.0/(2.0*c**2.0))+1.0+aa*np.exp(-(v_obs-bb)**2.0/(2.0*cc**2.0))

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[1548.202,1550.772],'abs_osc_strs':[0.19,0.0952], 'em_waves':[1548.202,1550.772],'em_osc_strs':[0.19,0.0952],'res':[True,True],'fluor':[False,False],'p_r':[1.0,1.0],'p_f':[0.0,0.0],'final_waves':[1548.202,1550.772],'line_num':[1,1], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### Mg II 2796.35,2803.53

v_obs = np.linspace(-1000,1500,1000)
lam_ref = 2796.35

shift = ((2803.53-2796.35)/(2796.35))*(2.99792458*10**5)
a,b,c = 1.0,0.0,75
aa,bb,cc = 1.0,shift,75

background = a*np.exp(-(v_obs-b)**2.0/(2.0*c**2.0))+1.0+aa*np.exp(-(v_obs-bb)**2.0/(2.0*cc**2.0))

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[2796.35,2803.53],'abs_osc_strs':[0.608,0.303], 'em_waves':[2796.35,2803.53],'em_osc_strs':[0.608,0.303],'res':[True,True],'fluor':[False,False],'p_r':[1.0,1.0],'p_f':[0.0,0.0],'final_waves':[2796.35,2803.53],'line_num':[1,1], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()

#### Fe II 343.49,2364.83,2380.76

v_obs = np.linspace(-1000,6000,2000)
lam_ref = 2343.49

background = np.ones_like(v_obs)

alpha,psi,gamma, tau, v_0, v_w, v_ap, f_c, k, delta = np.pi/2.0,0,1.0,1.0,25.0,500.0,500.0,1.0,0.0,3.0
OCCULTATION = True
APERTURE = True

flow_parameters = {'alpha':alpha, 'psi':psi, 'gamma':gamma, 'tau':tau, 'v_0':v_0, 'v_w':v_w, 'v_ap':v_ap, 'f_c':f_c, 'k':k, 'delta':delta}
profile_parameters = {'abs_waves':[2343.49],'abs_osc_strs':[.114,.0495,.0351], 'em_waves':[2343.49,2364.83,2380.76],'em_osc_strs':[.114,.0495,.0351],'res':[True,False,False],'fluor':[False,True,True],'p_r':[0.657794676807,0.657794676807,0.657794676807],'p_f':[0.22433460076+0.117870722433,0.22433460076,0.117870722433],'final_waves':[2343.49,2364.83,2380.76],'line_num':[3], 'v_obs':v_obs,'lam_ref':lam_ref, 'APERTURE':APERTURE,'OCCULTATION':OCCULTATION}

start = time.time()
yy = Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters)
end = time.time()
print(end - start)

plt.plot(v_obs,yy,'r')
plt.show()
