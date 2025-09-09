# Ablation Studies with hyper-parameter h
Ablation studies were conducted on the image–text dataset 2D Chest X-Ray
+ clinical report with the ResNetMLP model, investigating the effect of hyper-
parameter h. This specific dataset and model were chosen because they utilize the
modalities in a largely balanced manner. For both modalities, vision and text, we
used different combinations of hi and tracked the resulting modality contribution
values. In our setup we chose h(vision) ∈ {1, 4, 16, 49, 196, 256}, ranging from
whole-image occlusion (224 × 224) to patch occlusion (14 × 14), and h(text)
∈ {1, 2, 4, 8, 16, 32} ranging from occluding the entire text to a single word. For
instance, h(vision) = 49 indicates, that the occluded mask has size 32 × 32. Since
224/32 = 7 there are 7 patches along each spatial dimension, resulting in a total
of 7 · 7 = 49 patches.
Finally we plotted the resulting modality contributions m(vision) and m(text)
in a 3D plot (Fig. 5). The first two axes correspond to h(vision) and h(text), while
the third axis shows m(vision) and m(text). The rendered surface was generated
by interpolating 36 data points. In the Figs. 6 and 7 the results of the analysis
for selected 2D slices are plotted, representing the variation of the modality
contributions with respect to the hyper-parameters h(vision) and h(text).
