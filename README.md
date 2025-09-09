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

## Citation

If you use this work, please cite our paper:

**Gapp, C., Tappeiner, E., Welk, M., Fritscher, K., Gizewski, E.R., & Schubert, R. (2025).**  
*What are You Looking at? Modality Contribution in Multimodal Medical Deep Learning Methods.*  
Conference for Computer Assisted Radiology and Surgery (CARS) — (in press).  
[arXiv:2503.01904](https://doi.org/10.48550/arXiv.2503.01904)

You can also use the BibTeX entry:

```bibtex
@article{Gapp_MCI,
  title={What are You Looking at? {M}odality Contribution in Multimodal Medical Deep Learning Methods},
  author={Christian Gapp and Elias Tappeiner and Martin Welk and Karl Fritscher and Elke R. Gizewski and Rainer Schubert},
  year={2025},
  doi={10.48550/arXiv.2503.01904},
  note = "{Conference for Computer Assisted Radiology and Surgery (CARS) -- (in press)}"
}
```


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
