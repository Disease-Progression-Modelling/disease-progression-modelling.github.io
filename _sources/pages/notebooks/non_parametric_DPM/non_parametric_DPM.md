# GP Progression Model

The Gaussian Process Progression Model (GPPM) is a model of disease progression estimating long-term biomarkers’ trajectories across the evolution of a disease, from the analysis of short-term individual measurements. 
GPPM software has been first presented in the work [Lorenzi, NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/), and subsequently extended in the GPPM-DS presented in [Garbarino and Lorenzi, IPMI 2019](https://doi.org/10.1002/alz.12083) and [Garbarino and Lorenzi, NeuroImage 2021](https://www.sciencedirect.com/science/article/pii/S1053811921002573).

GPPM and GPPM-DS enable the following analyses: 

- [GPPM] reoconstruct the profile of biomarkers evolution over time, 
- [GPPM] quantify the subject-specific disease severity associated with the measurements each individual (missing observations are allowed),
- [GPPM] estimate the ordering of the biomarkers from normal to pathological stages,
- [GPPM-DS] specify prior hypothesis about the causal interaction between biomarkers,
- [GPPM-DS] data-driven estimation of the interaction parameters, 
- [GPPM-DS] data-driven comparison between different hypothesis to identify the most plausible interaction dynamics between biomarkers,
- [GPPM-DS] model personalisation to simulate and predict subject-specific biomarker trajectories,

````{grid}
:gutter: 2

```{grid-item-card}
**Getting started**
^^^
The software comes with a [simple installation](https://gitlab.inria.fr/epione/GP_progression_model_V2) and an easy interface. 

An example of the basic usage of GPPM on synthetic and real data is available here:

[[Basic GPPM tutorial](https://disease-progression-modelling.github.io/pages/notebooks/non_parametric_DPM/GPPM_basic.html)]

[[Jupyter notebook](https://github.com/Disease-Progression-Modelling/disease-progression-modelling.github.io/blob/master/pages/notebooks/non_parametric_DPM/GPPM_basic.ipynb)]

[[Colab notebook](https://colab.research.google.com/drive/1JcouPj4KzOC_klOa2uwRvNHVtdjEensz?usp=sharing)]

An example of GPPM-DS on synthetic and real data is available here:

[[Basic GPPM-DS tutorial](https://disease-progression-modelling.github.io/pages/notebooks/non_parametric_DPM/GPPM_DS.html)]

[[Jupyter notebook](https://github.com/Disease-Progression-Modelling/disease-progression-modelling.github.io/blob/master/pages/notebooks/non_parametric_DPM/GPPM_DS.ipynb)]

[[Colab notebook](https://colab.research.google.com/drive/1OA8X2vZIelb2cGdicYXfFKwPOoWCoISl?usp=sharing)]

````

```{note}
The source code is available on [GitLab](https://gitlab.inria.fr/epione/GP_progression_model_V2). 
The software is freely distributed for academic purposes. All the commercial rights are owned by Inria.
```


