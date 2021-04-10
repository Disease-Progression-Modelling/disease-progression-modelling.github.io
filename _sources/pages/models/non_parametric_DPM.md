# Gaussian Process Progression Model

## The problem
Longitudinal dataset of measurements from neurodegenerative studies generally lack of a well-defined temporal reference, since the onset of the pathology may vary across individuals according to genetic, demographic and environmental factors. Therefore, age or visit date information are biased time references for the individual longitudinal measurements. There is a critical need to define the AD evolution in a data-driven manner with respect to an absolute time scale associated to the natural history of the pathology.

<img src="../../_static/img/gppm/GPPM_intro1.png" width="640px" alt="The problem">

## The solution: GPPM ;)

The [Gaussian Process Progression Model (GPPM)](https://gitlab.inria.fr/epione/GP_progression_model_V2) is based on the probabilistic estimation of biomarkers’ trajectories and on the quantification of the uncertainty of the predicted individual pathological stage. The inference framework accounts for a time reparameterization function, encoding individual differences in disease timing and speed relative to the fixed effect. 

<img src="../../_static/img/gppm/animated.gif" width="400px" alt="GGPM in action">

Thanks to the probabilistic nature of GPPM, the resulting long-term disease progression model can be used as a statistical reference representing the transition from normal to pathological stages, thus allowing probabilistic diagnosis in the clinical scenario. The model can be further used to quantify the diagnostic uncertainty of individual disease severity, with respect to missing measurements, biomarkers, and follow-up information.

GPPM has three key components underlying its methodology: 1) it defines a non-parametric, Gaussian process, Bayesian regression model for individual trajectories, 2) introduces a monotonicity information to impose some regular behaviour on the trajectories, and 3) models individual time transformations encoding the information on the latent pathological stage.

## A variety of applications

The model was originally published in [NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/), and demonstrated on a large cohort of amyloid positive Alzheimer's disease individuals.

<img src="../../_static/img/gppm/gppm_AD.png" width="500px" alt="GPPM from (Lorenzi et al., Neuroimage 2017)">


GPPM has been extended in recent years, and is now capable of disentangling spatio-temporal disease trajectories from collections of [high-dimensional brain images](https://doi.org/10.1016/j.neuroimage.2019.116266)...

<img src="../../_static/img/gppm/full_brain.gif" width="600px" alt="GPPM from (Abi Nader et al, NeuroImage 2019)">

... and imposing a variety of [biology-inspired constraints on the biomarker trajectories](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_5).

<img src="../../_static/img/gppm/animation_sara.gif" width="600px" alt="GPPM from (Garbarino and Lorenzi, IPMI 2019)"> 


## Getting started:

- The code is available [here](https://gitlab.inria.fr/epione/GP_progression_model_V2). 

- Some scientific literature on the GPPM methods:
  - Modeling biomarkers' trajectories in Alzheimer's disease: [Lorenzi, et al., NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/)
  - Modeling the dynamics of amyloid propagation across brain networks: [Garbarino and Lorenzi, IPMI 2019](https://doi.org/10.1002/alz.12083), [Garbarino and Lorenzi, NeuroImage 2021](https://www.sciencedirect.com/science/article/pii/S1053811921002573)
  - Spatio-temporal analysis of multimodal changes from time series of brain images [Abi Nader, et al., NeuroImage 2020](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_5)
