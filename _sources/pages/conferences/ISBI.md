# ISBI 2021


```{figure} ../../_static/img/conferences/ISBI_2021.png
---
name: ISBI
align: center
width: 50%
---
**ISBI 2021**.
```


::::{grid}
:gutter: 2

:::{grid-item}
:column: col-12
:card: border-2 shadow
:header: bg-warning
**Workshop abstract**
^^^

We propose to organize a  3 - hours tutorial  on Neurological Disease Progression Modelling (DPM) to  present the most advanced and mature data - driven disease progression models and the related  software. Each model will be described and illustrated through interactive hands - on sessions.  

DPM of neurodegenerative disorders is an active research area aiming at reconstructing the  pathological evolution of neurodegenerative pathologies through the statistical analysis of  heterogeneous multi - modal biomedical information. DPM models have the potential to highlight the  temporal dynamics and relationship across biomarkers. Moreover, by automatically staging patients  along the disease time axis, they can be used as predictive tools for diagnostic purposes. The  fi eld has  blossomed in recent years thanks to the ava ilability of large biomedical  datasets, such as ADNI,  AIBL, and PPMI, as they offer the possibility of testing novel research and  modelling hypotheses.  Computational models constructed from such data sets lead to a deeper understanding of the complex pathophysiology of neurodegenerative diseases.

<!-- {badge}`Conference workshop,badge-primary`
{badge}`Tutorial,badge-primary`
{badge}`3-hours,badge-primary` -->
::::



[Link to the conference](https://biomedicalimaging.org/2021)


## Schedule
`Introduction to the symposium /` 10 min

`Discrete DPM / ` by _Neil Oxtoby & Vikram Venkatraghavan_ • 55 min

: Discrete models represent disease progression as a cumulative sequence in which biomarker abnormality occurs (disease “events”), together with uncertainty (positional variance) in that sequence. The most mature discrete DPM is the event-based model (Fonteijn-NeuroImage-2012; Young-Brain-2014; Venkatraghavan-NeuroImage-2019), which is able to infer a sequence from cross-sectional cohort data. Conceptually, this longitudinal picture of neurological disease progression is estimable because earlier events will have commensurately higher prevalence in a cohort containing a spectrum of clinical cases. Mathematically this is evaluated as more individuals having a higher data-driven likelihood of abnormality in the earlier events. With sufficient representation across combinations of abnormal and normal observations, the likelihood of any full ordered sequence can be estimated, and thus the most likely sequences can be revealed. The probabilistic sequence estimated by an event-based model is useful for state-of-the-art precision in fine-grained staging of individuals — assessment of an individual’s disease progression stage — by calculating the likelihood of their biomarker data, given the sequence. Software for the event-based model is part of a suite of models available from [Europond repository](https://github.com/EuroPOND/europond-software).

`Parametric DPM / ` by _Igor Koval_ • 55 min

: The generic model of (Schiratti-JMLR-2017) learns trajectories of changes from longitudinal manifold-valued (i.e. structured) observations. The model jointly estimates an average long-term history of changes, as well as its variability in terms of untangled static and dynamic components. Additionally, given a new patient, it is possible to estimate a personalized disease progression base on his individual biomarker. Such approach can be further specialized to model the progression of a single or several correlated biomarkers such as cognitive scores (Schiratti-NIPS-2015), surface or volume measurements such as the brain cortical thickness (Koval-MICCAI-2017), or the 3D geometry of objects such as segmented sub-cortical structures (Bône-2018-CVPR). The descriptive quality of this parametric approach is illustrated on [Digital-Brain](https://www.digital-brain.org) in the context of Alzheimer’s disease. With a pragmatic take based on the [Leaspy software](https://gitlab.com/icm-institute/aramislab/leaspy), interactive notebook-based demonstrations will gradually unveil how biomarkers enables us to estimate a long-term disease progression and predict the progression at the individual level.


`non-parametric DPM /` by _Sara Garbarino & Marco Lorenzi_ • 55 min

: The GP Progression Model of (Lorenzi-NeuroImage-2017) is a non-parametric probabilistic DPM which reconstructs long term pathological biomarkers trajectories from short clinical data and quantifies the variability associated with each trajectory. The model can be used for probabilistic diagnosis, as it assesses patients’ stages and diagnostic uncertainty in de-novo individuals. The GP Progression Model is [now available online](http://gpprogressionmodel.inria.fr/) with a simple and intuitive front-end. In this talk we will introduce and illustrate the model with an interactive demonstration of its performances on clinical data. We will provide to the participants a tutorial based on a Python notebook, so they can directly test and reproduce the tutorial material. During the tutorial we will also present the latest advances on the model which are currently being developed at the Epione team, INRIA Sophia Antipolis, including application to spatio-temporal modelling of protein dynamics (Garbarino-IPMI-2019), as well as extensions to the modeling of voxel-based imaging datasets (Abi Nader-NeuroImage-2019). [GP progression repository](https://gitlab.inria.fr/epione/GP_progression_model_V2).
