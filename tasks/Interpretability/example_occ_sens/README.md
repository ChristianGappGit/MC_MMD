## Chest X-RAY

![img1](ChestXRay_OccSens.png "ChestXRay OccSens")

![img2](ChestXRay_OccSens_text_CXR1897_IM-0581-1001_MEAN.png "ChestXRay_OccSens_text_CXR1897_IM-0581-1001_MEAN OccSens")

![img3](ChestXRay_OccSens_text_CXR1897_IM-0581-1001_MAX.png "ChestXRay_OccSens_text_CXR1897_IM-0581-1001_MAX OccSens")

Fig.: CXR1897_IM-0581-1001: Correctly predicted disease: support devices. Modality contribution vision : text = 0.24 : 0.76. Model: ViTLLAMA II. From blue to red the contribution (low to high) from a single patch (vision) or word (text) to the task is highlighted. Top, left to right: source image, GradCAM, class specific Occlusion Sensitivity for class support devices (MONAI), Occlusion Sensitivity averaged over all classes (CG, i.e. *ours*). The red patch in the upper right area in image Occ. sens. (MONAI) has the highest contribution to the class support devices. The same area is colored blue in image Occ. sens. (CG), as this patch has the lowest average contribution to all classes.  Bottom: Text. MEAN: The words *no* and *acute* have the highest average contribution, *catheter* has the lowest. MAX: *catheter* has the highest contribution to one class: support devices.


## BRSET

![img4](BRSET_OccSens.png "BRSET OccSens")

![img5](BRSET_OccSens_tabular_img03501_MEAN.png "BRSET_OccSens_tabular_img03501_MEAN OccSens")

![img6](BRSET_OccSens_tabular_img03501_MAX.png "BRSET_OccSens_tabular_img03501_MAX OccSens")

Fig.: img03501: Correctly predicted disease: drusens. Modality contribution vision : tabular = 0.95 : 0.05. Model: ResNetMLP. Importance (low to high) is colored from blue to red. Top, left to right: source image, GradCAM, class specific Occlusion Sensitivity for class drusens (MONAI), Occlusion Sensitivity averaged over all classes (CG, i.e. *ours*). Bottom: tabular data with attributes patient age, comorbidities, diabetes time, insulin use, patient sex, exam eye, diabetes from left to right. MEAN: The patient's age has the highest contribution, patient sex the lowest in average. MAX: patient's age is the most significant attribute for one class: drusens.
