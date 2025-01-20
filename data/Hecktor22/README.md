# MICCAI 2022 Grand Challenge Hecktor 2022

https://hecktor.grand-challenge.org/

## Download Data:

1) Register and Sign in (therein License Agreements must be made)

2) Go to Data Download (or open https://hecktor.grand-challenge.org/data-download-2/)

3) Follow the instructions and download the data


## Data Preprocessing:

The head and neck tumor segmentation training dataset contains data from 524 patients. For each patient 3D CT, 3D PET images and segmentation masks of the extracted tumors are provided together with clinical information in the form of tabular data. For the RFS (Relapse Free Survival) time prediction task, i.e. a regression problem, labels (0,1), indicating the occurrence of relapse and the RFS times (for label 0) and the PFS (Progressive Free Survival) times (label 1) in days are made available for 488 patients.
Due to some incomplete data and non-fitting segmentation masks, we finally could use image-tabular data pairs (CT segmentation mask + tabular data) of 444 patients. 

For this we overlayed the masks and CTs and extracted the intersections.
