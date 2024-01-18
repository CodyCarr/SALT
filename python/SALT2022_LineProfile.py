import numpy as np
from SALT2022_Absorption import computeABS
from SALT2022_Emission import computeEM
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
multiprocessing.set_start_method('fork')  
from functools import partial

def Function_CallABS(abs_waves,abs_osc_strs,v_obs,parameters,j):
    return computeABS(abs_waves[j],abs_osc_strs[j],v_obs,parameters[j])

def Function_CallEM(em_waves,em_osc_strs,lam_ref,v_obs,New_Flux_Lists,parameters,l):
    return computeEM(em_waves[l],em_osc_strs[l],lam_ref,v_obs,New_Flux_Lists[l],parameters[l]) 

def makeAbsorptionPROFILE(waves,lam_ref,v_obs,norm_flux,SALT_parameters,Absorption_Profiles):

    if SALT_parameters[0][6] == 0:
        return np.zeros_like(v_obs)

    NumberOfProfiles = len(waves)
    speed_of_light=2.99792458e5
    line_profiles = norm_flux

    for i in range(NumberOfProfiles):     
        #computes absorption profile centered on line 
        Absorption = Absorption_Profiles[i]
        #shifts absorption profile to correct observed velocity range in regards to lam_ref
        velocity_shift = (speed_of_light*(waves[i]-lam_ref)/lam_ref)
        idx1 = np.searchsorted(v_obs,velocity_shift, side="left")
        idx2 = np.searchsorted(v_obs,0, side="left")
        INDEX = idx1-idx2
        Absorption = np.roll(Absorption,INDEX)

        #computes final Absorption Profile and intermediate profiles 
        norm_flux = norm_flux + norm_flux*Absorption
        line_profiles = np.vstack((line_profiles,norm_flux))

    return line_profiles

def makeEmissionPROFILE(waves,lam_ref,v_obs,SALT_parameters,Emission_Profiles):

    if SALT_parameters[0][6] == 0:
        return np.zeros_like(v_obs)

    NumberOfProfiles = len(waves)
    speed_of_light=2.99792458e5    
    line_profiles = np.zeros(len(Emission_Profiles[0]))
    
    for i in range(NumberOfProfiles):
        
        #computes Emission profile centered on line 
        Emission = Emission_Profiles[i]
        #shifts Emission profile to correct observed velocity range in regards to lambda_ref
        vel_shift = (speed_of_light*(waves[i]-lam_ref)/lam_ref)
        idx1 = np.searchsorted(v_obs,vel_shift, side="left")
        idx2 = np.searchsorted(v_obs,0, side="left")
        INDEX = idx1-idx2
        Emission = np.roll(Emission,INDEX)

        #computes final Emission Profile 
        line_profiles += Emission

    return line_profiles

def Line_Profile(v_obs,lam_ref,background,flow_parameters,profile_parameters,profile_type):

    abs_waves,abs_osc_strs,em_waves,em_osc_strs,res,fluor,p_r,p_f,final_waves,line_num,v_obs, lam_ref, APERTURE, OCCULTATION = profile_parameters.values()
    alpha,psi,gamma,tau,v_0,v_w,v_ap,f_c,k,delta = flow_parameters.values()

    parameters_abs = np.tile(np.array([alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,delta,APERTURE],dtype= object), (len(abs_waves),1))
    parameters_em = np.tile(np.array([alpha,psi,gamma,tau,-1.0*v_0,-1.0*v_w,-1.0*v_ap,f_c,k,delta,APERTURE,False,False,OCCULTATION,1,0],dtype= object), (len(em_waves),1))

    parameters_em[:,11] = res
    parameters_em[:,12] = fluor
    parameters_em[:,14] = p_r
    parameters_em[:,15] = p_f
    
    with Pool(max_workers=len(abs_waves)) as inner_pool:
        Absorption_Profiles = list(inner_pool.map(partial(Function_CallABS,abs_waves,abs_osc_strs,v_obs,parameters_abs),range(len(abs_waves))))
        
    Normalized_Flux_Lists = makeAbsorptionPROFILE(abs_waves,lam_ref,v_obs,background,parameters_abs,Absorption_Profiles)

    New_Flux_Lists = np.array([])
    for i in range(len(line_num)):
        flux_list = np.tile(np.array(Normalized_Flux_Lists[i],dtype=object),(line_num[i],1))
        New_Flux_Lists = np.concatenate((New_Flux_Lists,flux_list)) if New_Flux_Lists.size else flux_list
        
    with Pool(max_workers=len(em_waves)) as inner_pool:
        Emission_Profiles = list(inner_pool.map(partial(Function_CallEM,em_waves,em_osc_strs,lam_ref,v_obs,New_Flux_Lists,parameters_em),range(len(em_waves))))
        
    Emission = makeEmissionPROFILE(final_waves,lam_ref,v_obs,parameters_em,Emission_Profiles)
    Absorption = Normalized_Flux_Lists[-1]

    if profile_type == 'absorption':
        return Absorption
    elif profile_type == 'emission':
        return Emission
    else:
        return Absorption+Emission

    return spectrum
