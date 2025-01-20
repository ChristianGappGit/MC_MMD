"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

SwinViT_torch + MLP (down transform) + tanh() for Vision
MLP + tanh() for Clinical Data
MLP for Fusion
Cls Head (dense + sigmoid) for Classification
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks.mlp import MLPBlock #created for vision, but usable for clinical (i.e. tabular here) data as well
from torchvision.models import SwinTransformer
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    """
    Classification layer.
    with Sigmoid activation function.
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.denseCLS = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        cls_output = self.denseCLS(first_token_tensor)
        cls_output = self.activation(cls_output)
        return cls_output

class SwinViTMLPNet_torch(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            spatial_dims: int,
            num_classes=1, 
            num_clinical_features=10,
            embed_dim=96,
            num_heads=[3,6,12,24],
            depths=[2, 2, 6, 2],
            window_size=[8,8],
            dropout_rate=0.1,
            act = "tanh",
            only_vision = False,
            only_clinical = False,
            ):
        super().__init__()

        self.only_vision = only_vision
        self.only_clinical = only_clinical
        self.multimodal = not only_vision and not only_clinical

        assert not (only_vision and only_clinical), "Only one of only_vision and only_clinical can be True. Set both to False in order to use multimodal model." 
        if embed_dim % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        num_vision_features = num_clinical_features #set to the same number of features

        #VISION
        if self.only_vision or self.multimodal:
            vision_model = SwinTransformer(
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=4.0,
                    dropout=dropout_rate,
                    attention_dropout=dropout_rate,
                    stochastic_depth_prob=dropout_rate,
                    num_classes=num_vision_features, #attention here !!!
                    #use deafault:
                    #norm_layer=nn.LayerNorm,
                    #block = None,
                   #downsample=PatchMerging,
                )
        if act == "tanh":
            self.vision_model = nn.Sequential(vision_model, nn.Tanh())
        elif act == "relu":
            self.vision_model = nn.Sequential(vision_model, nn.ReLU())
        else:
            self.vision_model = vision_model

        #CLINICAL
        if self.only_clinical or self.multimodal:
            clinical_mlp_dim = 4*num_clinical_features #times 4 is quite common
            self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.Tanh())

        #FUSION
        if self.multimodal:
            fusion_dim = num_clinical_features + num_vision_features #for concatenation of vision and clinical features
            fusion_mlp_dim = 4*fusion_dim
            self.fusion = MLPBlock(fusion_dim, fusion_mlp_dim, dropout_rate)

        if self.multimodal:
            self.classification_head = Classifier(fusion_dim, num_classes)
        elif self.only_vision:
            self.classification_head = Classifier(num_vision_features, num_classes)
        elif self.only_clinical:
            self.classification_head = Classifier(num_clinical_features, num_classes)

    def forward(self, clinical_info, img):
        if self.multimodal:
            vision_features = self.vision_model(img) #shape is [batch_size, ...
            tabular_features = self.clinical_model(clinical_info) #shape is [batch_size, num_features, clinical_mlp_dim]
            #print("vision_features.shape", vision_features.shape)
            #print("vision_features", vision_features)
            #print("tabular_features", tabular_features)
            #print("tabular_features.shape", tabular_features.shape)
            #concat the features
            mixed_features = torch.cat([vision_features, tabular_features], dim=1)
            #print("vision_features.shape", vision_features.shape)
            #print("tabular_features.shape", tabular_features.shape)
            #print("mixed_features.shape", mixed_features.shape)
            #print("mixed_features", mixed_features)
            x = self.fusion(mixed_features)
            #print("x.shape after fusion", x.shape)
            #print("x_out_fusion", x)
            x = self.classification_head(x)
            #print("x.shape after classification", x.shape)
            #print("x_output", x)
            return x
        elif self.only_vision:
            vision_features = self.vision_model(img)
            x = self.classification_head(vision_features)
            return x
        elif self.only_clinical:
            tabular_features = self.clinical_model(clinical_info)
            x = self.classification_head(tabular_features)
            return x