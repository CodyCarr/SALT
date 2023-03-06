Introduction
============

The semi-analyitcal line transfer (SALT) model is a semi-analytical radiation transfer model designed to predict the spectra of galactic outflows.  This documentation shows how to install and compute SALT.  Examples of different line profile predictions are provided.  In addition, we provide a detailed example showcasing how to fit SALT to a real spectrum.

About the Model
***************

The SALT model was first introduced by Scarlata and Panagia (2015), but has since been modified by Carr et al. (2018,2023).  The 
following documentation is based on the formalism presented in Carr et al. (2023).  While we refer the reader to this paper for 
the details regarding the calculation of the radiation transfer, we provide a physical description of the model and its parameter space here.  All projects which use the SALT model should cite the Carr et al. (2023) paper.

The basic model assumes a spherical source of isotropically emitted radiation which propogates through an expanding medium (i.e., an outflow).  The outflow is characterized by a density field, :math:`n(r)=n_0(\frac{r}{R_{SF}})^{-\delta}`, and velocity field, :math:`v(r)=v_0(\frac{r}{R_{SF}})^{\gamma}`.  The geometry of the outflow is that of a bicone described by an opening angle, :math:`\alpha`, and orientation angle, :math:`\psi`, which can open all the way into a sphere.  A picture of the general model is provided in Figure 1.  In addition, the outflow is assumed to be embedded in a spherical envelope of dust with density which scales with the density of the outflow (see Carr et al. 2021 for a description).  The observational effects of a limiting observing aperture is also considered.  

.. figure:: ../../images/f1.jpg
   :class: with-border

   The model consists of a biconical outflow of opening angle, :math:`\alpha`, and orientation angle, :math:`\psi`, which extends from the surface of the star forming region at radius, :math:`R_{SF}`, to a terminal radius, :math:`R_{W}`.

The SALT model represents a solution to the radiation transfer of resonant photons through the outflow.  In addition, SALT can handle fluorescent emission following resonant scattering (see Scarlata and Panagia 2015 for details).  The next section shows how to compute SALT given various line profiles.  Examples include lines with and without fluorescence.  

Free Parameters
***************

.. list-table:: SALT Parameters
   :widths: 25 100
   :header-rows: 1

   * - Parameter
     - Description
   * - :math:`\alpha`
     - half opening angle
   * - :math:`\psi`
     - orientation angle
   * - :math:`\tau`
     - optical depth divided by :math:`f_{ul}\lambda_{ul}\ [\text{Å}^{-1}]`
   * - :math:`\gamma`
     - velocity field power law index
   * - :math:`v_{0}`
     - launch velocity :math:`[\rm km\ s^{-1}]`
   * - :math:`v_{\infty}`
     - terminal velocity :math:`[\rm km\ s^{-1}]`
   * - :math:`\delta`
     - density field power law index
   * - :math:`\kappa`
     - dust opacity multiplied by :math:`R_{SF}n_{0,dust}`
   * - :math:`v_{ap}`
     - velocity field at :math:`R_{AP}`
   * - :math:`f_c`
     - covering fraction inside outflow

Using the Model
===============

The SALT model code consists of three python scripts: SALT2022_Absorption.py, SALT2022_Emission.py, and SALT2022_LineProfile.py.  All three scripts can be obtained from GitHub by entering the following command in a terminal window  git clone git@github.com:CodyCarr/SALT.git.  The model can also be accessed by downloading the three scripts from the following documentation.  Working from within the SALT folder, one can run SALT with Python using the following script.  

.. code-block:: python

    import numpy as np
    from SALT2022_LineProfile import Line_Profile

    # SALT parameters
    alpha,psi,gamma, tau, v_0, v_w, f_c, k, delta, v_ap = np.pi/2.0,0.0,1.0,1.0,25.0,500.0,1.0,0.0,3.0

    # Background to be scattered through SALT (this example assumes a flat continuum)
    background = np.ones_like(v_obs)

    # Turn Aperture and Occultation effect on or off (True or False)
    OCCULTATION = True
    APERTURE = True

    # refence wavelength
    lam_ref = 1193.28
    
    # Observed velocity range centered on lam_ref
    v_obs = np.linspace(-2000,2000,1000)

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

    
Examples
========

