import numpy as np

def makeAbsorptionPROFILE(wavelengths,oscillator_strengths,lambda_ref,v_obs,Normalized_Flux,SALT_parameters,Absorption_Profiles):

    if SALT_parameters[0][6] == 0:
        return np.zeros_like(v_obs)

    NumberOfProfiles = len(wavelengths)
    speed_of_light=2.99792458e5
    line_profiles = Normalized_Flux

    for i in range(NumberOfProfiles):
        
        #computes absorption profile centered on line 
        Absorption = Absorption_Profiles[i]
        #shifts absorption profile to correct observed velocity range in regards to lambda_ref
        velocity_shift = (speed_of_light*(wavelengths[i]-lambda_ref)/lambda_ref)
        MIN1 = min(v_obs, key=lambda x:abs(x-velocity_shift))
        MIN2 = min(v_obs, key=lambda x:abs(0-x))
        INDEX = np.where(np.isclose(v_obs,MIN1))[0]-np.where(np.isclose(v_obs,MIN2))[0]
        Absorption = np.roll(Absorption,INDEX)

        #computes final Absorption Profile and intermediate profiles 
        Normalized_Flux = Normalized_Flux + Normalized_Flux*Absorption
        line_profiles = np.vstack((line_profiles,Normalized_Flux))

    return line_profiles

def makeEmissionPROFILE(wavelengths,oscillator_strengths,lambda_ref,v_obs,Normalized_Flux,SALT_parameters,Emission_Profiles):

    if SALT_parameters[0][6] == 0:
        return np.zeros_like(v_obs)

    NumberOfProfiles = len(wavelengths)
    speed_of_light=2.99792458e5    
    line_profiles = np.zeros(len(Emission_Profiles[0]))
    
    for i in range(NumberOfProfiles):
        
        #computes Emission profile centered on line 
        Emission = Emission_Profiles[i]
        #shifts Emission profile to correct observed velocity range in regards to lambda_ref
        velocity_shift = (speed_of_light*(wavelengths[i]-lambda_ref)/lambda_ref)
        MIN1 = min(v_obs, key=lambda x:abs(x-velocity_shift))
        MIN2 = min(v_obs, key=lambda x:abs(0-x))
        INDEX = np.where(np.isclose(v_obs,MIN1))[0]-np.where(np.isclose(v_obs,MIN2))[0]
        Emission = np.roll(Emission,INDEX)

        #computes final Emission Profile 
        line_profiles += Emission

    return line_profiles
