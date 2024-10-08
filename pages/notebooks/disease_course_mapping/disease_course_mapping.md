# Disease Course Mapping with Leaspy
_by Igor Koval_

```{figure} ../../../_static/img/disease_course_mapping/logo.png
---
height: 150px
name: Logo
align: center
```


::::{grid}
:gutter: 2

:::{grid-item}
:column: col-12
:card: border-2 shadow
:header: bg-warning
**_Disease Course Mapping_ with Leaspy**
^^^

Leaspy is a Python software package that implements the [Disease Course Mapping](https://disease-progression-modelling.github.io/pages/models/disease_course_mapping.html) methods. In particular, it is designed for the statistical analysis of **longitudinal data**, particularly **medical** data that comes in a form of **repeated observations** of patients at different time-points.
Considering these series of short-term data, the software aims at :
- recombining them to reconstruct the long-term spatio-temporal trajectory of progression
- positioning each patient observations relatively to the group-average progression
- quantifying the impact of different cofactors (gender, genetic mutation, environmental factors, etc) on the progression of the biomarkers
- imputing missing values
- predicting future observations
- simulating virtual patients

<!-- {badge}`Software,badge-primary`
{badge}`Python package,badge-primary`
{badge}`Open source,badge-primary`
{badge}`Tutorials,badge-primary`
{badge}`Continuous integration,badge-primary` -->
::::

The software is distributed on [Gitlab](https://gitlab.com/icm-institute/aramislab/leaspy) under the GNU GPLv3 Licence. It has a complete [documentation](https://leaspy.readthedocs.io/en/latest/) as well as an active community that you can solicit through the use of the dedicated [bug & issue tracker](https://gitlab.com/icm-institute/aramislab/leaspy/-/issues). The software operates on <!--{badge}`Mac,badge-dark` and {badge}`Linux,badge-dark` - {badge}`Windows,badge-dark` is also working though no guarantee can be given as no specific development was developed for this platform.-->



```{note}
**Leaspy** originally comes from  from LEArning Spatiotemporal Patterns in Python.
```

## **Usage**

The package has been written to offer a user-friendly API in order to be used by non-experts. The API essentially includes the following functions

`leaspy.fit(...)`
: To estimate the average progression of the biomarkers

`leaspy.personalize(...)`
: To estimate the individual parameters that allows to derive the average progression to fit individual Data

`leaspy.estimate(...)`
: To impute or predict the values of an individual at any time-point

`leaspy.simulate(...)`
: To simulate synthetic patients

`leaspy.load(...)` & `leaspy.save(...)`
: To load and save the entire description of the disease progression, share it or reuse it on another cohort


```{tip}
**New developers are welcome to participate and contribute** !

While offering an easy-to-use API, the package has been designed with an internal modular architecture in order to make new developments possible. In particular, new developments are on they way to enlarge the possibilities of types of disease progression, in particular with ordinal data or progressions affected by drugs.
```


## **Tutorials**

We have developed few tutorials to better understand the goals of Disease Course Mapping, in particular how it allows to go beyond the current limitations of linear mixed-effects models, while getting familiar with Leaspy.


::::{grid}
:gutter: 2

:::{grid-item}
:column: col-8 offset-md-2
:header: bg-warning
:card: m-2 shadow
:body: text-justify

**Tutorial 1: Limitations of linear mixed-effects model**
^^^
This introduction to longitudinal data progressively unveils the limitations of linear models, beginning by a cross-patient linear regression, going to individual linear regressions, then to a linear mixed-effects model, finishing by the limitations of linear models that Leaspy is able to overcome.

[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/TP1_LMM.html)
+++
<!-- {badge}`1 hour,badge-warning` {badge}`longitudinal data,badge-primary` {badge}`Linear model,badge-primary` {badge}`Mixed effects model,badge-primary` -->

---

**Tutorial 2: `Hello World` with Leaspy**
^^^
This introduction to Leaspy gives a handful overview of Leaspy possibilities along with the user-friendly commands to use it

[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/TP2_leaspy_beginner.html)
+++
<!-- {badge}`30 minutes,badge-warning` {badge}`longitudinal data,badge-primary` {badge}`non-linear mixed-effects model,badge-primary` -->

---

**Tutorial 3: Real-case usage**
^^^
This tutorial is designed for the ones that want to use Leaspy with their own data. We therefore designed this tutorial to present step-by-step the different operations that you might go through : data manipulation, normalization, interpretation of the algorithm convergence, in-depth understanding of the parameters and hyperparameters, etc.

[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/TP3_advanced_leaspy.html)
+++
<!-- {badge}`1 hour 30,badge-warning` {badge}`longitudinal data,badge-primary` {badge}`Missing values,badge-primary` {badge}`longitudinal data,badge-primary` -->
::::


## **Installation**

The installation procedure is entirely detailed on the [dedicated Gitlab repository](https://gitlab.com/icm-institute/aramislab/leaspy)
