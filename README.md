# Project Overview

Parkinson's Disease (PD) is a prevalent neurodegenerative disorder affecting approximately  (percent) of adults nationwide, with its main symptoms being tremor, bradykinesia, etc. 
In addition to the condition lacking a cure, it is also difficult to diagnose during its early progression, commonly confused with similar parkinsonian disorders up into its final stages. . 

Among the many forms of diagnosis, this project aims to further leverage non-invasive modalities - spiral drawings and audio recordings - to detect PD, as compared to similar CV methods using MRI and PET scans.
However, the relevant datasets available comprise very small sample sizes (More information available in the _ Data Availability_ section).
To combat the issue of small datasets, misalignment between datasets, and limited datasets, we demonstrate the usefulness of a multimodal classification model using a pipeline comprising dataset-dependent feature extraction and a dataset-aware Mixture of Experts (MoE) model. 
The architecture boasts its robustness by being able to make a prediction given any combination of modalities, and generalizability

The key goal is to answer the following research questions:

- Can machine-learning be leveraged to accurately diagnose PD using only non-invasive data collection methods?
- Can data augmentation and Mixture of Experts address the issues of limited sample sizes and disjoint datasets?
- How can we define optimally performing pipelines tailored to each modality? Can our model achieve good performance even if one modality is lacking?

# Pipeline Overview
## Data Preprocessing
## Data Augmentation
## Mixture of Experts with Routing Layer
## Model Evaluation

## Future Works


# Manuscript Status


# Data Availability



# Quickstart Guide
## Requirements

# Manuscript Status

# Acknowledgements

This work was conucted as part of the University of Wyoming's HUMANS MOVE: NSF REU Site.



