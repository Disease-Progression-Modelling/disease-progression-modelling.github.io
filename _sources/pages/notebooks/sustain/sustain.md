# Disease Course Sequence Subtyping with Subtype and Stage Inference
_by Alex Young_

```{figure} ../../../_static/img/sustain.png
---
width: 100pc
name: SuStaIn schematic
align: center
```

````{grid}
:gutter: 2

```{grid-item-card}
**_Disease Course Sequence Subtyping_: Subtype and Stage Inference**
^^^

Subtype and Stage Inference (SuStaIn) is a generalisation of event-based modelling that adds clustering to discover multiple data-driven sequences of disease progression using **cross-sectional** data.

The software:

- constructs a subtype model of a chronic, progressive disease consisting of multiple pathophysiological cascades (fine-grained temporal sequences of events);
- stages and subtypes individuals within the model, representing cumulative abnormality along each subtype progression sequence;
- does this all probabilistically and without predefined biomarker cutpoints.

<!-- {badge}`Software,badge-primary`
{badge}`Python package,badge-primary`
{badge}`Open source,badge-primary`
{badge}`Tutorials,badge-primary` -->
````

The software for classical SuStaIn is distributed via the UCL POND group's [GitHub](https://github.com/ucl-pond) account.

The software should operate across operating systems, but specific requirements, e.g., python package versions, are detailed in each repository.

## **Usage**

The [pySuStaIn](https://github.com/ucl-pond/pySuStaIn) package includes user-friendly functions to perform key operations in the Disease Course Sequencing pipeline:

`pySuStaIn.ZscoreSustain: run_sustain_algorithm(...)`
: Converts multimodal biomarker data into event and subtype probabilities by fitting mixture models to patient/control data using Kernel Density Estimation (?).

`pySuStaIn.ZscoreSustain._plot_sustain_model`
: Plotting tools for visualizing model outputs`

## **Tutorial(s)**

We have developed an introductory tutorial to understand Disease Course Sequence Subtyping. We are planning to provide an example on real data from a publicly available dataset.

````{grid}

```{grid-item-card}
**Tutorial 1: SuStaIn and simulated data**
^^^
This introduction to Subtype and Stage Inference is a walkthrough where you will fit a subtype model using the pySuStaIn software on simulated data.

+++
[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/sustain/T1_sustain_walkthrough.html)
<!-- {badge}`30 minutes,badge-warning` {badge}`cross-sectional data,badge-primary` -->
```

```{grid-item-card}
**Tutorial 2: SuStaIn and real data**
^^^
This planned walkthrough invovles fitting a subtype model using the pySuStaIn software on real data. 

Probably data from [ADNI](https://adni.loni.usc.edu) (data will not be provided here).

+++
Tutorial link will go here
```

````