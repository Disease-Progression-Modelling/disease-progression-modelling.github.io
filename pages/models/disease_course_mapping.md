# Disease Course Mapping
_by Igor Koval_

````{panels}
:column: col-12
:card: border-2
:header: bg-warning
**_Disease Course Mapping_ in a nutshell**
^^^
Neurodegenerative disease progresses over periods of time longer than the usual individual observations and with an important inter-individual variability. It directly prevents to straightforwardly disentangle a reference disease progression from its manifestation at an individual level.

For that purpose, we have developed a technique called **Disease Course Mapping**. Its aim is to estimate, both spatially and temporally, a unified spectrum of disease progression from a population. It directly allows to :
- describe the average progression (along with its variability) of the biomarkers over the entire course of the disease
- position any individual onto this map to precisely characterize the patients trajectory and predict his current and future stages

{badge}`Longitudinal data,badge-primary`
{badge}`Non-linear Mixed-effects model,badge-primary`
{badge}`Description of disease progression,badge-primary`
{badge}`Personalized staging & prediction,badge-primary`

````

## A picture is worth a thousands words


```{figure} ../../_static/img/disease_course_mapping/trajectory_mapping.png
---
name: Disease Course Mapping
align: center
width: 100%
---
**Disease Course Mapping**. The central panel shows the progression of two biomarkers across the course of a neurodegenerative disease. Each curve represents the progression of these biomarkers over time. The central curve corresponds to the average progression while the envelop shows the variability of progression within the population. This average progression corresponds to the idealized blue and orange curves on the four subpanels. Each subpanel displays an individual progression that has been positionned onto the disease course map. The green progression are likely progressions. On the other hand, the red panels show progression that are not likely to been seen in the population.
```



## Model description

```{admonition} Input data
We here consider **longitudinal data**, i.e. patients with multiple visits along time, each included repeated observations : cognitive tests, neuropsychological assessments, medical imaging features, etc.
```

The objective is to describe the average progression of the different features, variables, biomarkers, ... For the sake of clarity, we consider two biomarkers that have been normalized such that a value of 0 corresponds to normal pre-symptomatic stages where 1 corresponds to pathological stages.

We consider that we have been able to charaterize the progression of these two biomarkers over a long period of time within two idealized logistic progression. Now, how do we relate the model to the the observation of a new individual that has been seen at four different visits?


```{figure} ../../_static/img/disease_course_mapping/model_explanation_1.png
---
name: Individual parameters
align: center
width: 100%
---

```

We consider that individual observations derive from the average biomarker progressions based on three parameters : the time-shift that accounts for the delay or advance of the disease, the acceleration factor that informs about the pace of the progression, and, the inter-marker spacing that controls for the individual-specific ordering of events.



```{figure} ../../_static/img/disease_course_mapping/calibration.png
---
name: Calibration
align: center
width: 50%
---
Calibration
```


```{figure} ../../_static/img/disease_course_mapping/model_explanation_2.png
---
name: Explanation 2
align: center
width: 100%
---
Explanation 2
```



```{figure} ../../_static/img/disease_course_mapping/prediction.png
---
name: prediction
align: center
width: 50%
---
Predictions
```



## References
More detailed explanations about the models themselves and  about the estimation procedure can be found in the following articles :

- **Mathematical framework**: *A Bayesian mixed-effects model to learn trajectories of changes from repeated manifold-valued observations*. Jean-Baptiste Schiratti, Stéphanie Allassonnière, Olivier Colliot, and Stanley Durrleman.  The Journal of Machine Learning Research, 18:1–33, December 2017. [Open Access PDF](https://hal.archives-ouvertes.fr/hal-01540367).
- **Application to imaging data**: *Statistical learning of spatiotemporal patterns from longitudinal manifold-valued networks*. I. Koval, J.-B. Schiratti, A. Routier, M. Bacci, O. Colliot, S. Allassonnière and S. Durrleman. MICCAI, September 2017. [Open Access PDF](https://arxiv.org/pdf/1709.08491.pdf)
- **Application to imaging data**: *Spatiotemporal Propagation of the Cortical Atrophy: Population and Individual Patterns*. Igor Koval, Jean-Baptiste Schiratti, Alexandre Routier, Michael Bacci, Olivier Colliot, Stéphanie Allassonnière, and Stanley Durrleman. Front Neurol. 2018 May 4;9:235. Open Access PDF
- - **Application to data with missing values**: *Learning disease progression models with longitudinal data and missing values*. R. Couronne, M. Vidailhet, JC. Corvol, S. Lehéricy, S. Durrleman
- **Intensive application for Alzheimer's Disease progression**: *Simulating Alzheimer’s disease progression with personalised digital brain models*, I. Koval, A. Bone, M. Louis, S. Bottani, A. Marcoux, J. Samper-Gonzalez, N. Burgos, B. Charlier, A. Bertrand, S. Epelbaum, O. Colliot, S. Allassonniere & S. Durrleman, Under review [Open Access PDF](https://hal.inria.fr/hal-01964821/file/SimulatingAlzheimer_low_resolution%20%281%29.pdf)
- www.digital-brain.org: website related to the application of the model for Alzheimer's disease

## To go further

- If you want to see what the model is capable of : [Digital-brain website](https://www.digital-brain.org) (using Chrome, no iOS), with the related [paper](https://hal.archives-ouvertes.fr/hal-01744538v3/document) (under revision)

- If you want to take a look at the code : [Gitlab repo](https://gitlab.com/icm-institute/aramislab/leaspy/)



## Contacts
See Igor Koval in Contributors
http://www.aramislab.fr/
