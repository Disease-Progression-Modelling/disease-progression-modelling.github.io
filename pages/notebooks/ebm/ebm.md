# Disease Course Sequencing with the Event Based Model
_by Neil Oxtoby and Vikram Venkatraghavan_

```{figure} ../../../_static/img/ebm.png
---
height: 300px
name: EBM schematic
align: center
```


````{panels}
:column: col-12
:card: border-2 shadow
:header: bg-warning
**_Disease Course Sequencing_ with Event-Based Modelling**
^^^

Event-Based Modelling is a class of mathematical models, with associated Python softwares, that estimate a quantitative signature of disease progression (a Disease Course Sequence) using either **cross-sectional** or longitudinal **medical** data.

The softwares:
- reconstructs the pathophysiological cascade (fine-grained temporal sequence of events) for a chronic, progressive disease
- stages individuals along this fine-grained Disease Course Sequence, representing their cumulative abnormality along the group-average progression
- does this all probabilistically and without predefined biomarker cutpoints

{badge}`Software,badge-primary`
{badge}`Python package,badge-primary`
{badge}`Open source,badge-primary`
{badge}`Tutorials,badge-primary`
````

The software is distributed via the UCL POND group's [GitHub](https://github.com/ucl-pond) account, typically under the MIT license. The software should operate across operating systems, but specific requirements, e.g., python package versions, are detailed in each repository.

## **Usage**

The [KDE EBM](https://github.com/ucl-pond/kde_ebm) package includes user-friendly functions to perform key operations in the Disease Course Sequencing pipeline:

`kde_ebm.mixture_model: fit_all_kde_models(...), fit_all_gmm_models(...)`
: Converts multimodal biomarker data into event probabilities by fitting mixture models to patient/control data using Kernel Density Estimation or Gaussian mixture modelling.

`kde_ebm.plotting`
: Plotting tools for visualizing model outputs, e.g., `mixture_model_grid()`, `mcmc_uncert_mat()`, `mcmc_trace()`, `stage_histogram()`

`kde_ebm.mcmc`
: Tools for Markov Chain Monte Carlo fitting of the EBM, including bootstrap cross-validation.

The [pyEBM](https://github.com/88vikram/pyEBM) toolbox can be used to fit a traditional EBM, or a Discriminative EBM.

## **Tutorial(s)**

We have developed an introductory tutorial to understand Disease Course Sequencing using Event-Based Modelling. In future, we will provide an example on real data from a publicly available dataset.

````{panels}
:column: col-8 offset-md-2
:header: bg-warning
:card: m-2 shadow
:body: text-justify

**Tutorial 1: KDE EBM `Hello World`: example EBM on simulated data**
^^^
This introduction to Event-Based Modelling is a walkthrough where you will fit an EBM using the KDE EBM software and simulated data.

[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/ebm/T1_kde_ebm_walkthrough.html)
+++
{badge}`30 minutes,badge-warning` {badge}`cross-sectional data,badge-primary`

---

**Tutorial 2: pyEBM `Hello World`: example DEBM on simulated data**
^^^
This introduction to Discriminative Event-Based Modelling is a walkthrough where you will fit a DEBM using the pyEBM software and simulated data.

FIXME: add Vikram's tutorial (`T2_pyEBM_walkthrough.ipynb`?)
<!-- [Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/ebm/T2_pyEBM_walkthrough.html) -->
+++
{badge}`30 minutes,badge-warning` {badge}`crosssectional data,badge-primary`

---

**Tutorial 3: KDE EBM: example usage on real medical data**
^^^
_Coming soon_.

---

**Tutorial 4: DEBM: example usage on real medical data**
^^^
_Coming soon_.

---

````

## **Installation**

KDE EBM installation is explained in the [GitHub repository](https://github.com/ucl-pond/kde_ebm)

pyEBM installation is via `pip`: `pip install pyebm`
