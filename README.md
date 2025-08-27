# Project Overview

Parkinson's Disease (PD) is a prevalent neurodegenerative disorder affecting approximately  (percent) of adults nationwide, with its main symptoms being tremor, bradykinesia, etc. 
In addition to the condition lacking a cure, it is also difficult to diagnose during its early progression, commonly confused with similar parkinsonian disorders up into its final stages. . 

Among the many forms of diagnosis, this project aims to further leverage non-invasive modalities - **spiral drawings and audio recordings** - to detect PD, as compared to similar CV methods using MRI and PET scans.
However, the relevant datasets available comprise very small sample sizes (More information in [Data Availability](#)).
To combat the issue of small datasets, misalignment between datasets, and limited datasets, we demonstrate the usefulness of a multimodal classification model using a pipeline comprising dataset-dependent feature extraction and a dataset-aware Mixture of Experts (MoE) model. 
The architecture boasts its robustness by being able to make a prediction given any combination of modalities, and generalizability

The key goal is to answer the following research questions:

- Can machine-learning be leveraged to accurately diagnose PD using only non-invasive data collection methods?
- Can data augmentation and Mixture of Experts address the issues of limited sample sizes and disjoint datasets?
- How can we define optimally performing pipelines tailored to each modality? Can our model achieve good performance even if one modality is lacking?

# Pipeline Overview
This study focuses on the latter methods and attempts to expand upon previous deep-learning feature extraction utilizing spiral drawings and audio recordings. We do so by devising a model architecture with a Mixture of Experts (MoE) layer that is compatible with multiple datasets of distinct modalities. We also draw upon and review previous literature's model training pipelines comprising data preprocessing, data augmentation, and feature extraction per modality. Finally, we compare our model's performance on all input combinations with the current state-of-the-art. Our methodology allows versatility with model inputs and creates a well-rounded, comprehensive model for PD classification.

It;'s recommended that 
## Data Preprocessing
### Tabular
For simplicity, we filter each participant's file to only include data from Static Spiral Test (SST). X, Y, Z, time, and pressure columns are extracted from each sample. With the context of at-home recordings in mind, we discard Grip Angle data as common stylus devices do not record tilt/stylus orientation. However, we may experiment with using it in the future if considering use in professional, clinical environments where high-end technology is available. 
We do notice that some images from the healthy patients had a run-off line, but we chose to use them as is.

### Image
The original spiral images are already centered. Additionally, each image is converted to greyscale, resized to 224x224, and normalized. 

\subsubsection{Audio Dataset}
Each audio file is resampled to 16KHz for the later feature extraction step, and amplitude is normalized between [-1.0, 1.0]


Each dataset is then split into 70/30 train/test groups. We use a set random seed for reproducability.


## Data Augmentation
## Mixture of Experts with Routing Layer
## Model Evaluation

## Future Works


# Manuscript Status


# Data Provenance and Availability
All datasets used were accessed from public databases. 

The [tabular dataset](https://archive.ics.uci.edu/dataset/358/improved+spiral+test+using+digitized+graphics+tablet+for+monitoring+parkinson+s+disease.) was collected by the Department of Neurology in Cerrahpasa Faculty of Medicine at Istanbul University \cite{tabulardataset}. For 25 PD and 15 control patients, three variations of a spiral drawing task were presented, and the following was recorded on a Wacom Pen tablet:
1. x coordinate
2. y coordinate
3. z coordinate
4. Pressure on the screen with digital pen (0-1023 scale)
5. Grip angle of the individuals'
6. System time in ms that sample is recorded
7. Test ID (0: Static Spiral Test (SST), 1: Dynamic Spiral Test (DST), 2: Stability Test on Certain Point (STCP)

The [image dataset](https://www.kaggle.com/datasets/kmader/parkinsons-drawings/) comprises handwriting samples of what is essentially the Static Spiral Test (SST) task in the tabular dataset, only represented in drawing form. The source has pre-divided them into train/test folders, but we created our own from the entire dataset of 51 PD and 51 control patients.  

The [audio dataset](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127) includes voice recordings in .wav format with participant age and sex. For 40 PD and 41 control participants, each sample records a prolonged enunciation of the vowel /a/. They were collected using participants' telephones.  


# Quickstart Guide
## Requirements

# Manuscript Status

# Acknowledgements

This work was conucted as part of the University of Wyoming's HUMANS MOVE: NSF REU Site.



