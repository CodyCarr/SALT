import math as m
import numpy as np
from scipy.integrate import romb,cumtrapz
from scipy.optimize import root,brentq
from scipy.special import hyp2f1
from cmath import atan
from cmath import sqrt
catan = np.vectorize(atan)
csqrt = np.vectorize(sqrt)
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial, reduce
import operator
import warnings
warnings.filterwarnings("ignore")

def get_Lshell(y,Gamma2,Gamma3,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum):

    length = len(Normalized_Flux)-1
    xlow = y*np.cos(np.arcsin(y**(-Gamma3)))
    xhigh = y
    dx = x_spectrum[1]-x_spectrum[0]
    xmin = x_spectrum[0]
    xmax = x_spectrum[-1]
    
    CDF_x_sq = (x_spectrum)**2.0*(Normalized_Flux)
    CDF_x_sq = cumtrapz(CDF_x_sq,x_spectrum)
    CDF_x_sq = np.concatenate([[0], CDF_x_sq])
    CDF_x = cumtrapz(Normalized_Flux,x_spectrum)
    CDF_x = np.concatenate([[0], CDF_x])
    
    il = ((xlow-xmin)/(xmax-xmin)*(length)).astype(int)
    ih = ((xhigh-xmin)/(xmax-xmin)*(length)).astype(int)
    ihp1 = np.array(ih+1)
    ilp1 = np.array(il+1)
    
    ihp1[ihp1 > length] = length
    ilp1[ilp1 > length] = length
       
    CDF_H = Gamma2*y**Gamma10*CDF_x[ih]+Gamma2*Gamma9*y**Gamma11*CDF_x_sq[ih]
    CDF_Hp1 = Gamma2*y**Gamma10*CDF_x[ihp1]+Gamma2*Gamma9*y**Gamma11*CDF_x_sq[ihp1]
    CDF_L = Gamma2*y**Gamma10*CDF_x[il]+Gamma2*Gamma9*y**Gamma11*CDF_x_sq[il]
    CDF_Lp1 = Gamma2*y**Gamma10*CDF_x[ilp1]+Gamma2*Gamma9*y**Gamma11*CDF_x_sq[ilp1]

    term1 = CDF_H - CDF_Lp1
    term2 = ((xhigh-x_spectrum[ih])/dx) * (CDF_Hp1 - CDF_H)
    term3 = ((x_spectrum[ilp1]-xlow)/dx) * (CDF_Lp1 - CDF_L)
    L_Shell = term1 + (term2 + term3)
    
    return L_Shell 

def dust(y,x,y_inf,k,delta,Gamma3,Gamma5,Gamma12,Gamma13,Gamma14,Gamma15,Gamma16):
    #computes dust optical depth

    T_obs = np.where(x/y<=-1,m.pi,np.where(x/y>1,0,np.arccos(x/y)))

    z = y**Gamma3*np.sin(T_obs)

    #fixes a floating point error
    z = np.where(z < 10e-10,0.0,z)
    
    if delta != 1.0:
        tau_d = np.where(z == 0,k*Gamma5*(Gamma13-y**Gamma14),k*(((Gamma12*np.cos(np.arcsin((z/(y_inf**Gamma3)))))/(z)**Gamma16)*hyp2f1(0.5,Gamma15,1.5,-(Gamma12*np.cos(np.arcsin((z/(y_inf**Gamma3)))))**(2.0)/z**2.0) - ((y**(Gamma3)*np.cos(T_obs))/(z)**Gamma16)*hyp2f1(0.5,Gamma15,1.5,-(y**(Gamma3)*np.cos(T_obs))**(2.0)/z**2.0)))
        tau_d = np.where(tau_d<0,0,tau_d)
    else:
        tau_d = np.where(z == 0,k*np.log((y_inf/y)**Gamma3),k*(((Gamma12*np.cos(np.arcsin((z/(y_inf**Gamma3)))))/(z)**Gamma16)*hyp2f1(0.5,Gamma15,1.5,-(Gamma12*np.cos(np.arcsin((z/(y_inf**Gamma3)))))**(2.0)/z**2.0) - ((y**(Gamma3)*np.cos(T_obs))/(z)**Gamma16)*hyp2f1(0.5,Gamma15,1.5,-(y**(Gamma3)*np.cos(T_obs))**(2.0)/z**2.0)))
        tau_d = np.where(tau_d<0,0,tau_d)
    return tau_d

def getBernoulli3(y,tau,Gamma4,Gamma9):
    
    a = tau*y**Gamma4
    
    if Gamma9 != 0.0:
        Bernoulli3 = np.real(.5*(2.0-(((3.0**(.5)+1j)*a*catan((2.0*csqrt(Gamma9))/((-1j*a/3.0**(.5))+a+4.0)**(.5)))/(csqrt(Gamma9)*(12.0+(3.0-(3.0)**(.5)*1j)*a)**(.5)))-(((3.0**(.5)-1j)*a*catan((2.0*csqrt(Gamma9))/((1j*a/3.0**(.5))+a+4.0)**(.5)))/(csqrt(Gamma9)*(12.0+(3.0+(3.0)**(.5)*1j)*a)**(.5)))))
    else:
        Bernoulli3 = 1.0/(1.0+a/2.0+a**2.0/12.0)
        
    return Bernoulli3

def getAsymptotic(y,tau,Gamma4,Gamma9):

    return (3+Gamma9)/(3*tau*y**Gamma4)

def getBeta(y,tau,Gamma4,Gamma9,Gamma17,CHANGE_APPROX):

    Small_Approx = getBernoulli3(y,tau,Gamma4,Gamma9)
    Large_Approx = getAsymptotic(y,tau,Gamma4,Gamma9)

    if CHANGE_APPROX == 0:
        if Gamma17 > 1:
            return Large_Approx
        else:
            return Small_Approx
    else:
        return np.where(y < CHANGE_APPROX,getAsymptotic(y,tau,Gamma4,Gamma9),getBernoulli3(y,tau,Gamma4,Gamma9))

def get_Y_1_Root(y,x,Gamma18):
    
	y_1_poly = y**2.0*(1.0-y**Gamma18)-x**2.0
    
	return y_1_poly

def get_Y_1AP_Root(y,x,y_ap,Gamma2,Gamma7):
	y_1_poly_ap = y**Gamma7*x**2.0+y_ap**Gamma2-y**Gamma2
	return y_1_poly_ap

def get_Emission_Profile_GIV(y,x,tau,f_holes,A,G,H,O,P,R,Gamma1,Gamma2,Gamma3,Gamma4,Gamma8,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum):

    # arrays of same size as y ... 
    L_shell = get_Lshell(y,Gamma2,Gamma3,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum)
    Theta_C = np.arcsin(y**(-Gamma3))  
    COS_T = np.cos(Theta_C)
    THETA_AVG = (Theta_C+np.cos(Theta_C)*y**(-Gamma3))/(2.0*Theta_C)
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma9)*THETA_AVG)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 > y**Gamma3*P) & (x*y**Gamma1 < y**Gamma3*R)
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ...
        p = ((x*ytemp**Gamma1-ytemp**Gamma3*P)/(H))
        u = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-p)**2.0)**(.5)
        d = ytemp**Gamma3*O-(p*G)
        f_g = Gamma8*np.arctan((u)/(d))

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)
    
    return s1

def get_Emission_Profile_GIII(y,x,tau,f_holes,A,D,E,H,M,N,Gamma1,Gamma2,Gamma3,Gamma4,Gamma7,Gamma8,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum):

    # arrays of same size as y ... 
    L_shell = get_Lshell(y,Gamma2,Gamma3,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum)
    Theta_C = np.arcsin(y**(-Gamma3))  
    COS_T = np.cos(Theta_C)
    THETA_AVG = (Theta_C+np.cos(Theta_C)*y**(-Gamma3))/(2.0*Theta_C)
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma9)*THETA_AVG)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 >= y**Gamma3*D ) & (x*y**Gamma1 < y**Gamma3*N)
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E)
        k = ((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5)
        f_g = Gamma8*np.arctan((p)/(DD))

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # second condition
    cond = (x*y**Gamma1 < y**Gamma3*D)
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E)
        k = ((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5)
        f_g_u = Gamma8*np.arctan((p)/(DD))
        k = ((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5)
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E)
        f_g_l = Gamma8*np.arctan((p)/(DD))
        f_g = f_g_l + f_g_u

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)
    
    return s1
                
def get_Emission_Profile_GII(y,x,tau,f_holes,A,E,F,H,I,K,O,P,R,Gamma1,Gamma2,Gamma3,Gamma4,Gamma7,Gamma8,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum):

    # arrays of same size as y ... 
    L_shell = get_Lshell(y,Gamma2,Gamma3,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum)
    Theta_C = np.arcsin(y**(-Gamma3))  
    COS_T = np.cos(Theta_C)
    THETA_AVG = (Theta_C+np.cos(Theta_C)*y**(-Gamma3))/(2.0*Theta_C)
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma9)*THETA_AVG)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 >= y**Gamma3*R)
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        f_g = 1.0

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # extract y values on which we'll work into ytemp
    # second condition
    cond = ((x*y**Gamma1 > y**Gamma3*I+y**Gamma3*K*E) & (x*y**Gamma1 < y**Gamma3*R))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        n = (ytemp**Gamma3*I+ytemp**Gamma3*O*E)
        v = ((x*ytemp**Gamma1 - n)*F)
        ee = np.arccos(v/(ytemp**Gamma2-x**2.0*ytemp**Gamma7)**(.5))
        f_g = 1.0 - ee*Gamma8

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # extract y values on which we'll work into ytemp
    # third condition
    cond = ((x*y**Gamma1 > y**Gamma3*P) & (x*y**Gamma1 < y**Gamma3*I+y**Gamma3*O*E))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ...
        b = ((x*ytemp**Gamma1-ytemp**Gamma3*I)/(H))
        d = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-b)**(2.0))**(.5)
        h = (ytemp**Gamma3*O-((x*ytemp**Gamma1 - ytemp**Gamma3*I)/(E)))
        f_g = Gamma8*np.arctan((d)/(h))

        # an array of size y. (only fill indices where third condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    return s1

def get_Emission_Profile_GI(y,x,tau,f_holes,A,C,D,E,H,M,N,Q,Gamma1,Gamma2,Gamma3,Gamma4,Gamma7,Gamma8,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum):

    # arrays of same size as y ... 
    L_shell = get_Lshell(y,Gamma2,Gamma3,Gamma9,Gamma10,Gamma11,Normalized_Flux,x_spectrum)
    Theta_C = np.arcsin(y**(-Gamma3))  
    COS_T = np.cos(Theta_C)
    THETA_AVG = (Theta_C+np.cos(Theta_C)*y**(-Gamma3))/(2.0*Theta_C)
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma9)*THETA_AVG)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 > y**Gamma3*N)
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        f_g = 1.0

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # extract y values on which we'll work into ytemp
    # second condition
    cond = ((x*y**Gamma1 > y**Gamma3*(D)) & (x*y**Gamma1 > C*(y**Gamma3*M-y**Gamma3*(D)/(C))) & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond):
        ytemp = y[cond]  

        # work with ytemp only ... 
        w = ytemp**Gamma3*M-ytemp**Gamma3*(D)/(C)
        v = (x*ytemp**Gamma1 - (w*C))*Q
        ee = np.arccos(v/(ytemp**Gamma2-x**2.0*ytemp**Gamma7)**(.5))
        f_g = 1.0 - ee*Gamma8

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # third condition
    cond = ((x*y**Gamma1 > y**Gamma3*(D)) & (x*y**Gamma1 < C*(y**Gamma3*M-y**Gamma3*(D)/(C)))  & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E)
        k = ((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5)
        f_g = Gamma8*np.arctan((p)/(DD))

        # an array of size y. (only fill indices where third condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # fourth condition
    cond = ((x*y**Gamma1 < y**Gamma3*(D)) & (x*y**Gamma1>= C*(y**Gamma3*M-y**Gamma3*(D)/(C))) & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        k = ((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5)
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E)
        f_g_l = Gamma8*np.arctan((p)/(DD))

        w = ytemp**Gamma3*M-(ytemp**Gamma3*(D)/(C))
        v = (x*ytemp**Gamma1 - (w*C))*Q
        ee = np.arccos(v/(ytemp**Gamma2-x**(2.0)*ytemp**Gamma7)**(.5))
        f_g_u = 1.0 - ee*Gamma8

        f_g = f_g_u + f_g_l

        # an array of size y. (only fill indices where fourth condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)

    # fifth condition
    cond = ((x*y**Gamma1 < y**Gamma3*(D)) & (x*y**Gamma1 < C*(y**Gamma3*M-y**Gamma3*(D)/(C))))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ... 
        k = ((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5)
        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E)
        f_g_l = Gamma8*np.arctan((p)/(DD))

        DD = (ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E)
        k = ((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H))
        p = ((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5)
        f_g_u = Gamma8*np.arctan((p)/(DD))

        f_g = f_g_u+f_g_l

        # an array of size y. (only fill indices where fifth condition is True)
        s1[cond] = f_holes*f_g*L_shell[cond]*(1.0-np.exp(-tau_0[cond]))/(2.0*ytemp)
    
    return s1

#sets up the integrals for the blue shifted emission component 
def Blue_Emission_Integral(x,alpha,psi,y_inf,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX):
    
    lower_bound = max(x,1.0)

    #aperture 
    if PROFILE[0] == True:
        if y_inf**GAMMA[6]*x**2.0+SALT[5]**GAMMA[1]-y_inf**GAMMA[1] < 0:
            upper_bound = brentq(get_Y_1AP_Root,abs(x),y_inf,args =(x,SALT[5],GAMMA[1],GAMMA[6]))
        else:
            upper_bound = y_inf
    else:
        upper_bound = y_inf

    if  upper_bound <= lower_bound:
        I2 = 0.0
    else:
        y_range = np.linspace(lower_bound,upper_bound,513)
        y_range[0]=y_range[0]+.000001
        delta_y = (upper_bound-lower_bound)/513.0
        if alpha +psi > m.pi/2.0 and psi-alpha <= 0.0:    
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GI(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GI(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I2 = romb(np.exp(-tau_d)*get_Emission_Profile_GI(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha <= 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I2 = romb(np.exp(-tau_d)*get_Emission_Profile_GII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha + psi > m.pi/2.0 and psi-alpha > 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GIII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GIII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I2 = romb(np.exp(-tau_d)*get_Emission_Profile_GIII(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha >= 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GIV(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I2 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GIV(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I2 = romb(np.exp(-tau_d)*get_Emission_Profile_GIV(y_range,x,SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        else:
            I2 = 0


    return I2

#sets up the integrals for the red shifted emission component
def Red_Emission_Integral(x,alpha,psi,y_inf,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX):

    #Occultation
    if PROFILE[3] == True:
        if y_inf**2.0*(1.0-y_inf**GAMMA[17])-x**2.0 > 0:
            lower_bound = brentq(get_Y_1_Root,abs(x),y_inf,args =(x,GAMMA[17]))
        else:
            lower_bound = y_inf
    else:
        lower_bound = max(abs(x),1.0)

    #Aperture
    if PROFILE[0] == True:
        if y_inf**GAMMA[6]*x**2.0+SALT[5]**GAMMA[1]-y_inf**GAMMA[1] < 0:
            upper_bound = brentq(get_Y_1AP_Root,abs(x),y_inf,args =(x,SALT[5],GAMMA[1],GAMMA[6]))
        else:
            upper_bound = y_inf
    else:
        upper_bound = y_inf
    
    if upper_bound <= lower_bound:
        I3 = 0.0
    else:
        y_range = np.linspace(lower_bound,upper_bound,513)
        y_range[0]=y_range[0]+.000001
        delta_y = (upper_bound-lower_bound)/513.0
        if alpha +psi > m.pi/2.0 and psi-alpha <= 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GI(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GI(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I3 = romb(np.exp(-tau_d)*get_Emission_Profile_GI(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha <= 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I3 = romb(np.exp(-tau_d)*get_Emission_Profile_GII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha + psi > m.pi/2.0 and psi-alpha > 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GIII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GIII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I3 = romb(np.exp(-tau_d)*get_Emission_Profile_GIII(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[6],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha >= 0.0:
            if PROFILE[1] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_R = BB*p_r/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_R*get_Emission_Profile_GIV(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            elif PROFILE[2] == True:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                BB = getBeta(y_range,SALT[3],GAMMA[3],GAMMA[8],GAMMA[16],CHANGE_APPROX)
                F_F = p_f/(1.0-p_r*(1.0-BB))
                I3 = romb(np.exp(-tau_d)*F_F*get_Emission_Profile_GIV(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
            else:
                tau_d = dust(y_range,x,y_inf,SALT[7],SALT[8],GAMMA[2],GAMMA[4],GAMMA[11],GAMMA[12],GAMMA[13],GAMMA[14],GAMMA[15])
                I3 = romb(np.exp(-tau_d)*get_Emission_Profile_GIV(y_range,abs(x),SALT[3],SALT[6],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[7],GAMMA[8],GAMMA[9],GAMMA[10],Normalized_Flux,x_spectrum),delta_y)
        else:
            I3 = 0

    return I3

def CallFunction(x_array,y_inf,alpha,psi,tau_0,v_0,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX,j):
    xvalues = x_array[j]
    I_list_E=[]
    for x in xvalues:
        if x == 0.0:
            x = 1.0e-10		
        if abs(x) > abs(y_inf) or alpha == 0 or tau_0 == 0.0:
            I_list_E.append(0.0)
        elif x>0:
            I_new_E = Blue_Emission_Integral(x,alpha,psi,y_inf,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX)
            I_list_E.append(I_new_E)
        else:
            I_new_E = Red_Emission_Integral(x,alpha,psi,y_inf,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX)
            I_list_E.append(I_new_E)
    return I_list_E

def computeEM(wavelength,oscillator_strength,lambda_ref,v_obs,Normalized_Flux,parameters):
    #input: APERTURE,RESONANCE,FLUORESCENCE -->True or False; v_obs,parameters,Gauss,Normalized_Flux -->lists; everything else-->float
    #output: list
    
    alpha,psi,gamma,tau,v_0,v_w,v_ap,f_holes,k,delta,APERTURE,RESONANCE,FLUORESCENCE,OCCULTATION,p_r,p_f = parameters
    tau_0 = wavelength*oscillator_strength*tau

    speed_of_light=2.99792458e5
    velocity_shift = -(speed_of_light*(wavelength-lambda_ref)/lambda_ref)
    MIN1 = min(v_obs, key=lambda x:abs(x-velocity_shift))
    MIN2 = min(v_obs, key=lambda x:abs(0-x))
    INDEX = np.where(np.isclose(v_obs,MIN1))[0]-np.where(np.isclose(v_obs,MIN2))[0]
    Normalized_Flux = np.roll(Normalized_Flux,INDEX)
    x_spectrum = v_obs/v_0

    #Compute floats to define Geometry here
    A = m.sin(alpha)
    B = m.cos(alpha-abs(psi-alpha))
    C = m.tan(alpha-abs(psi-alpha))
    D = m.sin(psi+alpha-m.pi/2.0)
    E = m.tan(psi)
    F = m.tan(m.pi/2.0-psi)
    G = m.cos(psi)
    H = m.sin(psi)
    I = m.cos(psi+alpha)
    J = m.cos(m.pi/2.0-psi)
    K = m.sin(psi+alpha)
    L = m.sin(alpha-abs(psi-alpha))
    M = m.cos(psi+alpha-m.pi/2.0)
    N = m.cos(abs(psi-alpha))
    O = m.sin(alpha+psi)
    P = m.cos(alpha+psi)
    Q = m.tan((m.pi/2.0-alpha+abs(psi-alpha)))
    R = m.cos(psi-alpha)
    S = m.cos(m.pi/2.0-psi)

    y_inf = v_w/v_0
    y_ap = v_ap/v_0

    #Compute floats to define kinematics here
    Gamma1 = ((1.0-gamma)/gamma)
    Gamma2 = (2.0/gamma)
    Gamma3 = (1.0/gamma)
    Gamma4 = ((1.0-(delta+gamma))/gamma)
    if delta != 1.0:
        Gamma5 = 1.0/(1-delta)
    else:
        Gamma5 = 1.0 #just a value holder (not used in code)
    Gamma6 = ((gamma-1.0)/gamma)
    Gamma7 = (2.0*(1.0-gamma)/gamma)
    Gamma8 = (1.0/m.pi)
    Gamma9 = (gamma-1.0)
    Gamma10 = ((2.0-gamma)/gamma)
    Gamma11 = ((2.0-3.0*gamma)/gamma)
    Gamma12 = y_inf**(1.0/gamma)
    Gamma13 = y_inf**((1.0-delta)/gamma)
    Gamma14 = (1.0-delta)/gamma
    Gamma15 = delta/2.0
    Gamma16 = 2.0+gamma
    Gamma17 = tau_0*y_inf**Gamma4
    Gamma18 = -2.0/gamma

    y_total = np.linspace(1.0,y_inf,100)
    Diff = np.array(getBernoulli3(y_total,tau_0,Gamma4,Gamma9))-np.array(getAsymptotic(y_total,tau_0,Gamma4,Gamma9))
    max_ind = argrelextrema(Diff,np.greater)
    
    if len(max_ind[0]) == 0:
        CHANGE_APPROX = 0
    else:
        CHANGE_APPROX = y_total[max_ind[0][0]] 

    GAMMA = [Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,Gamma9,Gamma10,Gamma11,Gamma12,Gamma13,Gamma14,Gamma15,Gamma16,Gamma17,Gamma18]
    GEOMETRY = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S]
    SALT = [alpha,psi,gamma,tau_0,y_inf,y_ap,f_holes,k,delta]
    PROFILE = [APERTURE,RESONANCE,FLUORESCENCE,OCCULTATION]

    x_blue = v_obs[np.where(v_obs<0)]/v_0
    x_red = v_obs[np.where(v_obs>=0)]/v_0
    x_array = np.array([x_blue,x_red])

    #the aperture of the telescope is closed
    if v_ap == 0:
        return np.zeros_like(v_obs)

    with Pool(max_workers=2) as inner_pool:
        Emission_Profiles = list(inner_pool.map(partial(CallFunction,x_array,y_inf,alpha,psi,tau_0,v_0,p_f,p_r,SALT,GAMMA,GEOMETRY,PROFILE,Normalized_Flux,x_spectrum,CHANGE_APPROX),range(2)))
    return reduce(operator.iconcat, Emission_Profiles, [])
            

