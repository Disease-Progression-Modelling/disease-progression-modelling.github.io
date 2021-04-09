# GP Progression Model



The optimization of GPPM iterates over two steps:

- *Estimation of biomarkers trajectories*. This is a regression problem, and is solved in GPPM by fitting a Gaussian Process (GP) mixed effect model to the observations. Gaussian processes are a fabulous family of non-parametric functions that can be used to solve a large variety of machine learning problems. In the case of GPPM, we impose that the trajectory must be monotonic over time, to describe steady evolutions from normal to pathological states.
- 
- *Estimation of subject-specific time reparameterization function*. This step idetifies the most likely instant during the pathological evolution at which the individual has been observed, by optimizing the timing of the measurements with respect to the trajectories estimated above. The current version of the model support the estimation of both time-shift and scaling parameters.
