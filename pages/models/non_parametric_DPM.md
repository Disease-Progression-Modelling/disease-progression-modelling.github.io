# Gaussian Process Progression Model

<img src="../../_static/img/gppm_AD.png" width="320px" alt="GPPM from (Lorenzi et al., Neuroimage 2017)">

## Background
The Gaussian Process Progression Model (GPPM) reformulates disease progression modelling within a probabilistic setting to quantify the diagnostic uncertainty of individual disease severity, with respect to missing measurements, biomarkers, and follow-up information. The model was originally published in [NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/) by Marco Lorenzi and colleagues, and there demonstrated on a large cohort of amyloid positive testing, Alzheimer's disease individuals.

GPPM has three key components underlying its methodology: 1) it defines a non-parametric, Gaussian process, Bayesian regression model for individual trajectories, 2) introduces a monotonicity information to impose some regular behaviour on the trajectories, and 3) models individual time transformations encoding the information on the latent pathological stage.

GPPM has been generalised in recent years, and is now capable of disentangling spatio-temporal disease trajectories from collections of [high-dimensional brain images](https://doi.org/10.1016/j.neuroimage.2019.116266), and imposing a variety of [biology-inspired constraints on the biomarker trajectories](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_5).


## Getting started:

- About the code.

- Some scientific papers about the GPPM method:
  - [Lorenzi, et al., NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/)
  - [Garbarino, et al., IPMI 2019](https://doi.org/10.1002/alz.12083)
  - [Abi Nader, et al., NeuroImage 2020](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_5)

