"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

DenseNet + MLP (down transform) + tanh() for Vision
MLP + tanh() for Clinical Data
MLP for Fusion
Cls Head (dense + sigmoid) for Classification
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks.mlp import MLPBlock #created for vision, but usable for clinical (i.e. tabular here) data as well
from monai.networks.nets import DenseNet121 as DenseNetMonai
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

class DenseMLPNet(nn.Module):   #memory problems as stride and kernel size cannot be set from here
    def __init__(
            self, 
            in_channels: int,
            img_size: Union[Sequence[int], int],
            spatial_dims: int,
            num_classes=1, 
            num_clinical_features=10,
            dropout_rate=0.1,
            pretrained_vision_net = False,
            act = "tanh",
            only_vision = False,
            only_clinical = False
            ):
        super().__init__()
        
        self.only_vision = only_vision
        self.only_clinical = only_clinical
        self.multimodal = not only_vision and not only_clinical

        #VISION
        if self.only_vision or self.multimodal:
            self.vision_model = DenseNetMonai(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_clinical_features, #for later concatenation
                pretrained=pretrained_vision_net
            )
            #VISION Transform. not needed any more, as realiszed with cls token in DenseNet
            #transform the output of the vision model to the same shape as the clinical model
            #vision_features = hidden_size_vision*sequence_len_vision
            num_vision_features = num_clinical_features #set to the same number of features
            #self.vision_transform = nn.Linear(vision_features, num_vision_features)

        #CLINICAL
        if self.only_clinical or self.multimodal:
            clinical_mlp_dim = 4*num_clinical_features #times 4 is quite common
            if act == "tanh":
                self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.Tanh())
            elif act == "relu":
                self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.ReLU())
            else:
                self.clinical_model = MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate)

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
            vision_features = self.vision_model(img) #shape is [batch_size, sequence_len, hidden_size_vision]
            tabular_features = self.clinical_model(clinical_info) #shape is [batch_size, num_features, clinical_mlp_dim]
            #print("vision_features.shape", vision_features.shape)
            #print("vision_features", vision_features)
            #print("tabular_features", tabular_features)
            #print("tabular_features.shape", tabular_features.shape)
            #transform the vision features to the same shape as the clinical features
            #first: flatten the vision features, was [1,sequence_len,hidden_size_vision] now [1,sequence_len*hidden_size_vision]
            
            #the following two line are not relavant anymore, as the cls token is used ant hence output dim already matches the clinical features
            #vision_features = vision_features.flatten(1)
            #vision_features = self.vision_transform(vision_features) #transform to the same shape as the clinical features with a linear layer (MLP)
            
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