# MC_MMD

---

Modality Contribution in Multimodal Medical Datasets (MC_MMD)

---

"What are You looking at? Modality Contribution in Multimodal Medical Deep Learning Methods"

## method
The modality contribution is a measure for the importance of one modality in a multimodal dataset.
*(Details in paper)*.
We applied our method to three tasks.

## data
The data can be download with the instructions in data/*..

## tasks
* BRSET
* ChestXRay
* Hecktor22

## citation
Please cite:

*(will be added)*

## abstract

**Purpose.**
High dimensional, multimodal data can nowadays be analyzed by huge deep neural networks with little effort. Several fusion methods for bringing together different modalities have been developed. Particularly, in the field of medicine with its presence of high dimensional multimodal patient data, multimodal models characterize the next step. However, what is yet very underexplored is how these models process the source information in detail.

**Methods.**
To this end, we implemented an occlusion-based both model and performance agnostic modality contribution method that quantitatively measures the importance of each modality in the dataset for the model to fulfill its task. 
We applied our method to three different multimodal medical problems for experimental purposes.

**Results.**
Herein we found that some networks have modality preferences that tend to unimodal collapses, while some datasets are imbalanced from the ground up. Moreover, we could determine a link between our metric and the performance of single modality trained nets.

**Conclusion.**
The information gain through our metric holds remarkable potential to improve the development of multimodal models and the creation of datasets in the future. With our method we make a crucial contribution to the field of interpretability in deep learning based multimodal research and thereby notably push the integrability of multimodal AI into clinical practice. Our code is publicly available at https://github.com/ChristianGappGit/MC_MMD.
