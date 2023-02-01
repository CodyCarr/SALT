import math as m
import numpy as np
from scipy.integrate import romb
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

def get_Y_1_Root(y,x,Gamma10):
	y_1_poly = y**2.0*(1.0-y**Gamma10)-x**2.0  
	return y_1_poly

def get_Y_1AP_Root(y,x,y_ap,Gamma2,Gamma5):
	y_1_poly_ap = y**Gamma5*x**2.0+y_ap**Gamma2-y**Gamma2
	return y_1_poly_ap

def get_Absorption_Profile_GIV(y,x,tau,f_holes,A,G,H,O,P,R,Gamma1,Gamma2,Gamma3,Gamma4,Gamma6,Gamma7,Gamma8,Gamma9):

    # arrays of same size as y ... 
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma7)*(x/y)**2.0)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = ((x*y**Gamma1 > y**Gamma3*P) & (x*y**Gamma1 < y**Gamma3*R))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ...
        p = (((x*ytemp**Gamma1-ytemp**Gamma3*P)/(H)))
        u = (((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-p)**2.0)**(.5))
        d = (ytemp**Gamma3*O-(p*G))
        f_c = (Gamma6*np.arctan((u)/(d)))

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))
    
    return s1

def get_Absorption_Profile_GIII(y,x,tau,f_holes,A,D,E,H,M,N,Gamma1,Gamma2,Gamma3,Gamma4,Gamma6,Gamma7,Gamma8,Gamma9): 

    # arrays of same size as y ... 
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma7)*(x/y)**2.0)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = ((x*y**Gamma1 >= y**Gamma3*D ) & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond):
        ytemp = y[cond]

        # work with ytemp only ...
        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E))
        k = (((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5))
        f_c = (Gamma6*np.arctan((p)/(DD)))

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # second condition
    cond = (x*y**Gamma1 < y**Gamma3*D)
    if np.any(cond):
        ytemp = y[cond]  

        # work with ytemp only ...
        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E))
        k = (((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5))
        f_c_u = (Gamma6*np.arctan((p)/(DD)))

        k = (((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5))
        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E))
        f_c_l = (Gamma6*np.arctan((p)/(DD)))

        f_c = (f_c_l + f_c_u)

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    return s1

def get_Absorption_Profile_GII(y,x,tau,f_holes,A,E,F,H,I,K,O,P,R,Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,Gamma9):

    # arrays of same size as y ... 
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma7)*(x/y)**2.0)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 >= y**Gamma3*R)
    if np.any(cond):    
        ytemp = y[cond]

        # work with ytemp only ... 
        f_c = 1.0

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # second condition
    cond = ((x*y**Gamma1 > y**Gamma3*I+y**Gamma3*K*E) & (x*y**Gamma1 < y**Gamma3*R))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ...
        n = (ytemp**Gamma3*I+ytemp**Gamma3*O*E)
        v = ((x*ytemp**Gamma1 - n)*F)
        ee = np.arccos(v/(ytemp**Gamma2-x**2.0*ytemp**Gamma5)**(.5))
        f_c = 1.0 - ee*Gamma6

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # third condition
    cond = ((x*y**Gamma1 > y**Gamma3*P) & (x*y**Gamma1 < y**Gamma3*I+y**Gamma3*O*E))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ...  
        b = ((x*ytemp**Gamma1-ytemp**Gamma3*I)/(H))
        d = ((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-b)**(2.0))**(.5)
        h = (ytemp**Gamma3*O-((x*ytemp**Gamma1 - ytemp**Gamma3*I)/(E)))
        f_c = Gamma6*np.arctan((d)/(h))

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    return s1

def get_Absorption_Profile_GI(y,x,tau,f_holes,A,C,D,E,H,M,N,Q,Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,Gamma9):  

    # arrays of same size as y ... 
    tau_0 = (tau*y**Gamma4)/(1.0+(Gamma7)*(x/y)**2.0)
    s1 = np.zeros_like(y)

    # extract y values on which we'll work into ytemp
    # first condition
    cond = (x*y**Gamma1 > y**Gamma3*N)
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ... 
        f_c = 1.0

        # an array of size y. (only fill indices where first condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # second condition
    cond = ((x*y**Gamma1 > y**Gamma3*(D)) & (x*y**Gamma1 > C*(y**Gamma3*M-y**Gamma3*(D)/(C))) & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ... 
        w = (ytemp**Gamma3*M-ytemp**Gamma3*(D)/(C))
        v = ((x*ytemp**Gamma1 - (w*C))*Q)
        ee = (np.arccos(v/(ytemp**Gamma2-x**2.0*ytemp**Gamma5)**(.5)))
        f_c = (1.0 - ee*Gamma6)

        # an array of size y. (only fill indices where second condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # third condition
    cond = ((x*y**Gamma1 > y**Gamma3*(D)) & (x*y**Gamma1 < C*(y**Gamma3*M-y**Gamma3*(D)/(C)))  & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ... 
        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E))
        k = (((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5))
        f_c = (Gamma6*np.arctan((p)/(DD)))

        # an array of size y. (only fill indices where third condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # fourth condition
    cond = ((x*y**Gamma1 < y**Gamma3*(D)) & (x*y**Gamma1 >= C*(y**Gamma3*M-y**Gamma3*(D)/(C))) & (x*y**Gamma1 < y**Gamma3*N))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ... 
        k = (((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H)))
        p =(((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5))
        DD =((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E))
        f_c_l = (Gamma6*np.arctan((p)/(DD)))

        w = (ytemp**Gamma3*M-(ytemp**Gamma3*(D)/(C)))
        v = ((x*ytemp**Gamma1 - (w*C))*Q)
        ee = (np.arccos(v/(ytemp**Gamma2-x**(2.0)*ytemp**Gamma5)**(.5)))
        f_c_u = (1.0 - ee*Gamma6)

        f_c = (f_c_u + f_c_l)

        # an array of size y. (only fill indices where fourth condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))

    # extract y values on which we'll work into ytemp
    # fifth condition
    cond = ((x*y**Gamma1 < y**Gamma3*(D)) & (x*y**Gamma1 < C*(y**Gamma3*M-y**Gamma3*(D)/(C))))
    if np.any(cond): 
        ytemp = y[cond]

        # work with ytemp only ... 
        k = (((ytemp**Gamma3*D-x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**(2.0)-(ytemp**Gamma3*A-k)**2.0)**(.5))
        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)+x*ytemp**Gamma1/E))
        f_c_l = (Gamma6*np.arctan((p)/(DD)))

        DD = ((ytemp**Gamma3*M-(ytemp**Gamma3*D/E)-x*ytemp**Gamma1/E))
        k = (((ytemp**Gamma3*D+x*ytemp**Gamma1)/(H)))
        p = (((ytemp**Gamma3*A)**2.0-(ytemp**Gamma3*A-k)**2.0)**(.5))
        f_c_u = (Gamma6*np.arctan((p)/(DD)))

        f_c = (f_c_u+f_c_l)

        # an array of size y. (only fill indices where fifth condition is True)
        s1[cond] = f_holes*f_c*Gamma2*(ytemp**Gamma8+Gamma7*x**2.0*ytemp**(Gamma9))*(1.0-np.exp(-tau_0[cond]))
    
    return s1

#sets up the integrals for the aborption component 
def Absorption_Integral(x,alpha,psi,y_inf,SALT,GAMMA,GEOMETRY,APERTURE):
    #Aperture
    if (APERTURE == True) and (SALT[7] < 1.0):
        scale = SALT[7]**GAMMA[9]
        if y_inf**GAMMA[4]*x**2.0+SALT[7]**GAMMA[1]-y_inf**GAMMA[1] < 0:
            upper_bound = brentq(get_Y_1AP_Root,abs(x),y_inf,args =(x,SALT[7],GAMMA[1],GAMMA[4]))
        else:
            upper_bound = y_inf        
    else:
        scale = 1.0
        if y_inf**2.0*(1.0-y_inf**GAMMA[9])-x**2.0 > 0:
            upper_bound = brentq(get_Y_1_Root,abs(x),y_inf,args =(x,GAMMA[9]))
        else:
            upper_bound = y_inf

    lower_bound = max(x,1.0)

    if upper_bound <= lower_bound or x <= 0.0 or upper_bound == 0 or alpha == 0:
        I1 = 0.0
    else:
        y_range = np.linspace(lower_bound,upper_bound,513)
        y_range[0]=y_range[0]+.000001
        delta_y = (upper_bound-lower_bound)/513.0
            
        if alpha +psi > m.pi/2.0 and psi-alpha <= 0.0:
            I1 = romb(scale*get_Absorption_Profile_GI(y_range,x,SALT[3],SALT[5],GEOMETRY[0],GEOMETRY[2],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GEOMETRY[16],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[4],GAMMA[5],GAMMA[6],GAMMA[7],GAMMA[8]),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha <= 0.0:
            I1 = romb(scale*get_Absorption_Profile_GII(y_range,x,SALT[3],SALT[5],GEOMETRY[0],GEOMETRY[4],GEOMETRY[5],GEOMETRY[7],GEOMETRY[8],GEOMETRY[10],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[4],GAMMA[5],GAMMA[6],GAMMA[7],GAMMA[8]),delta_y)
        elif alpha + psi > m.pi/2.0 and psi-alpha > 0.0:
            I1 = romb(scale*get_Absorption_Profile_GIII(y_range,x,SALT[3],SALT[5],GEOMETRY[0],GEOMETRY[3],GEOMETRY[4],GEOMETRY[7],GEOMETRY[12],GEOMETRY[13],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[5],GAMMA[6],GAMMA[7],GAMMA[8]),delta_y)
        elif alpha +psi <= m.pi/2.0 and psi-alpha >= 0.0:
            I1 = romb(scale*get_Absorption_Profile_GIV(y_range,x,SALT[3],SALT[5],GEOMETRY[0],GEOMETRY[6],GEOMETRY[7],GEOMETRY[14],GEOMETRY[15],GEOMETRY[17],GAMMA[0],GAMMA[1],GAMMA[2],GAMMA[3],GAMMA[5],GAMMA[6],GAMMA[7],GAMMA[8]),delta_y)
        else:
            I1 = 0

    return I1

def computeABS(wavelength,oscillator_strength,v_obs,parameters):
    #input: v_obs,parameters-->lists; everything else-->float
    #output: list

    alpha,psi,gamma,tau,v_0,v_w,v_ap,f_holes,delta,APERTURE = parameters
    tau_0 = wavelength*oscillator_strength*tau

    I_list_A=[]

    A = m.sin(alpha)
    C = m.tan(alpha-abs(psi-alpha))
    D = m.sin(psi+alpha-m.pi/2.0)
    E = m.tan(psi)
    F = m.tan(m.pi/2.0-psi)
    G = m.cos(psi)
    H = m.sin(psi)
    I = m.cos(psi+alpha)
    K = m.sin(psi+alpha)
    M = m.cos(psi+alpha-m.pi/2.0)
    N = m.cos(abs(psi-alpha))
    O = m.sin(alpha+psi)
    P = m.cos(alpha+psi)
    Q = m.tan((m.pi/2.0-alpha+abs(psi-alpha)))
    R = m.cos(psi-alpha)

    B = 0
    J = 0
    L = 0
    S = 0

    y_inf = v_w/v_0
    y_ap = v_ap/v_0

    Gamma1 = ((1.0-gamma)/gamma)
    Gamma2 = (2.0/gamma)
    Gamma3 = (1.0/gamma)
    Gamma4 = ((1.0-(delta+gamma))/gamma)
    Gamma5 = (2.0*(1.0-gamma)/gamma)
    Gamma6 = (1.0/m.pi)
    Gamma7 = (gamma-1.0)
    Gamma8 = ((2.0-gamma)/gamma)
    Gamma9 = ((2.0-3.0*gamma)/gamma)
    Gamma10 = -2.0/gamma

    GAMMA = [Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6,Gamma7,Gamma8,Gamma9,Gamma10]
    GEOMETRY = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S]
    SALT = [alpha,psi,gamma,tau_0,y_inf,f_holes,delta,y_ap]

    for i in range(len(v_obs)):

        x = v_obs[i]/v_0

        if x == 0.0:
            #fixes floating point errors caused when x = 0
            x = 1.0e-10

        if abs(x) > abs(y_inf) or alpha == 0 or y_ap == 0:
            I_list_A.append(0.0)
        elif x>0:
            I_new_A = -Absorption_Integral(x,alpha,psi,y_inf,SALT,GAMMA,GEOMETRY,APERTURE)
            I_list_A.append(I_new_A)
        else:
            I_list_A.append(0.0)

    return I_list_A



        
