# Disease Course Mapping with Leaspy

**Disease Course Mapping** corresponds to the method of estimating the ....
Such methods, models and estimation algorithms have been developed in the **Leaspy** software package.

```{figure} ../../../_static/img/disease_course_mapping/logo.png
---
height: 150px
name: Logo
align: center
```

## Description
Leaspy, that originatelly comes from LEArning Spatiotemporal Patterns in Python, is a software package for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.
Considering these series of short-term data, the software aims at :
- recombining them to reconstruct the long-term spatio-temporal trajectory of evolution
- positioning each patient observations relatively to the group-average timeline, in term of both temporal differences (time shift and acceleration factor) and spatial differences (diffent sequences of events, spatial pattern of progression, ...)
- quantifying impact of cofactors (gender, genetic mutation, environmental factors, ...) on the evolution of the signal
- imputing missing values
- predicting future observations
- simulating virtual patients to unbias the initial cohort or mimis its characteristics

## Tutorials

The software package can be used with scalar multivariate data whose progression can be modeled by a logistic shape, an exponential decay or a linear progression.
The simplest type of data handled by the software are scalar data: they correspond to one (univariate) or multiple (multivariate) measurement(s) per patient observation.
This includes, for instance, clinical scores, cognitive assessments, physiological measurements (e.g. blood markers, radioactive markers) but also imaging-derived data that are rescaled, for instance, between 0 and 1 to describe a logistic progression.


There is three progression tutorials :
- linear mixed effect models
- leaspy TP2_leaspy_beginner
- leaspy advanced


## Repo for installation & support

The package is distributed under the GNU GENERAL PUBLIC LICENSE v3.

## Documentation
https://leaspy.readthedocs.io/en/latest/
