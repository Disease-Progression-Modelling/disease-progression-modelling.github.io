# Disease Course Sequencing and Phenotyping with the Subtype and Stage Inference Model
_by Alex Young_

```{figure} ../../../_static/img/sustain.png
---
height: 700px
name: SuStaIn schematic
align: center
```

````{panels}
:column: col-12
:card: border-2 shadow
:header: bg-warning
**_Disease Course Sequencing and Phenotyping_ with Subtype and Stage Inference model**
^^^

 Subtype and Stage Inference (SuStaIn) is a class of mathematical models, with associated Python softwares, that estimates simultaneously a quantitative signature of disease progression (a Disease Course Sequence) and disease subtypes (a Disease Phenotype) using **cross-sectional** **medical** data.

The softwares:
- reconstruct the pathophysiological cascade (fine-grained temporal sequence of events) and unravel the different subtypes of a chronic, progressive disease
- stages and subtypes individuals along this fine-grained Disease Course Sequence and Phenotypes, representing their cumulative abnormality along the sub-group-average progression
- does this all probabilistically and without predefined biomarker cutpoints

{badge}`Software,badge-primary`
{badge}`Python package,badge-primary`
{badge}`Open source,badge-primary`
{badge}`Tutorials,badge-primary`
````

The software for classical SuStaIn is distributed via the UCL POND group's [GitHub](https://github.com/ucl-pond) account, typically under the MIT license.

The software should operate across operating systems, but specific requirements, e.g., python package versions, are detailed in each repository.

## **Usage**

The [SuStaIn](https://github.com/ucl-pond/pySuStaIn) package includes user-friendly functions to perform key operations in the Disease Course Sequencing pipeline:

`pySuStaIn.ZscoreSustain: run_sustain_algorithm(...)`
: Converts multimodal biomarker data into event and subtype probabilities by fitting mixture models to patient/control data using Kernel Density Estimation (?).

`pySuStaIn.ZscoreSustain._plot_sustain_model`
: Plotting tools for visualizing model outputs`

## **Tutorial(s)**

We have developed an introductory tutorial to understand Disease Course Sequencing and Phenotyping using Subtype and Stage Inference model. In future, we will provide an example on real data from a publicly available dataset.

````{panels}
:column: col-8 offset-md-2
:header: bg-warning
:card: m-2 shadow
:body: text-justify

**Tutorial 1: SuStaIn tutorial using simulated data**
^^^
This introduction to Subtype and Stage Inference model is a walkthrough where you will fit an SuStaIn using the pySuStaIn software and simulated data.

[Go to the tutorial](https://disease-progression-modelling.github.io/pages/notebooks/sustain/T1_sustain_walkthrough.html)
+++
{badge}`30 minutes,badge-warning` {badge}`cross-sectional data,badge-primary`

---
