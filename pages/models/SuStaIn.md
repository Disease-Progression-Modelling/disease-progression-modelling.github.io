# SuStaIn

<img src="../../_static/img/sustain.png" width="720px" alt="SuStaIn from (Young et al. 2018)">

## Background

The Subtype and Stage Inference model (SuStaIn) of disease progression was invented by Alex Young. It was first published [in this Nature Communications 2018 article](https://doi.org/10.1038/s41467-018-05892-0) where it demostraded its utility for uncovering the hetereogenity in genetic Frontotemporal Dementia (FTD) and sporadic Alzheimer's disease (AD). Since then it has been used to study protein spreading patterns in neurodegenerative diseases like AD ([tau](https://doi.org/10.1038/s41591-021-01309-6) and [beta amyloid](https://www.neurology.org/doi/10.1212/WNL.0000000000200148)), Lewy Body dementia ([alpha synuclein](https://doi.org/10.1038/s41467-024-49402-x), LBD) and [TDP-43](https://doi.org/10.1093/brain/awad145) related proteinopathies. 

SuStaIn is uniquely capable of unravelling both temporal and phenotypic heterogeneity to identify population subgroups with common patterns of disease progression using only cross-sectional data --- a single visit per patient. This model represented a siginificant improvement over its predecesors, which focused only on temporal hetereogenity (stage-only models like [EBM](https://disease-progression-modelling.github.io/pages/models/event_based_model.html)) or phenotipic hetereogenity (subtype-only models), and assumptions like "the data is from a single subtype" or "all the patients are in the same stage" were needed. 

The concept behind SuStaIn is a generalisation from its predecesor (EBM): the data represents more than one trajectory.
