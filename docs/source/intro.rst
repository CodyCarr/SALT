Introduction
============

The semi-analyitcal line transfer (SALT) model is a semi-analytical 
radiation transfer model designed to predict the spectra of galactic outflows.  
This documentation shows how to install and compute spectra for arbitrary 
line profiles with SALT.  

About the Model
***************

The SALT model was first introduced by Scarlata and Panagia (2015), but has since been modified by Carr et al. (2018,2023).  The 
following documentation is based on the formalism presented in Carr et al. (2023).  While we refer the reader to this paper for 
the details, we provide a physical description of the model parameter space here.  All projects which use the SALT model should 
cite the Carr et al. (2023) paper.

SALT assumes a spherical source of isotropically emitted radiation which propogates through an expanding medium (i.e., an outflow).  
The outflow is characterized by a density field :math:`n_0(\frac{r}{R_{SF}})^{-\delta}`.
