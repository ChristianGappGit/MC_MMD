"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

ViT + MLP (down transform) + tanh() for Vision
MLP + tanh() for Clinical Data
MLP for Fusion
Cls Head (dense + sigmoid) for Classification
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks.mlp import MLPBlock #created for vision, but usable for clinical (i.e. tabular here) data as well
from monai.networks.nets import ViT
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
    
class TabularEmbedding(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.tabular_embedding = nn.Embedding(num_features, embedding_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, embedding_size))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward_embedding(self, x):
        #print("x_shape", x.shape)
        x = self.tabular_embedding(x)
        x = x + self.position_embedding
        # x_shape is [batch_size, x[1].shape=seq_len, embedding_size]
        #transform to [batch_size, embedding_size] with AdaptiveAvgPool1d
        #print("x.shape afer embedding", x.shape)
        x = x.transpose(1,2)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        return x
    
    def forward(self, x):
        return self.forward_embedding(x)
    
class ArgPass(nn.Module):
    """
    this is a dummy class to pass the first argument to the next layer
    for ViT
    Input: x (= Tuple of 2 elements: x[0], x[1])
    Output: x0
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[0]

class ViTMLPNet(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            spatial_dims: int,
            num_classes=1, 
            num_clinical_features=10,
            hidden_size_vision=768,
            mlp_dim=3072,
            num_heads=12,
            num_vision_layers=6,
            dropout_rate=0.1,
            qkv_bias=False,
            use_pretrained_vit = False,
            apply_tabular_embedding=False,
            tabular_embedding_size=96,
            num_embeddings_tabular = 32000, #default for llamaII
            only_vision = False,
            only_clinical = False,
            act = "tanh",
            ):
        super().__init__()

        self.only_vision = only_vision
        self.only_clinical = only_clinical
        self.multimodal = not only_vision and not only_clinical

        assert not (only_vision and only_clinical), "Only one of only_vision and only_clinical can be True. Set both to False in order to use multimodal model."
        
        #VISION Transform
        num_vision_features = num_clinical_features if not apply_tabular_embedding else tabular_embedding_size #set to the same number of features
        if self.only_vision or self.multimodal:
            vision_model = ViT(
                    in_channels = in_channels ,
                    img_size = img_size,
                    patch_size = patch_size,
                    hidden_size = hidden_size_vision,
                    mlp_dim = mlp_dim,
                    num_layers = num_vision_layers,
                    num_heads = num_heads,
                    pos_embed = "conv",
                    classification = True, #Important
                    num_classes = num_vision_features, #Attention here !!!
                    dropout_rate = dropout_rate,
                    spatial_dims = spatial_dims,
                    post_activation=None, #importan to set None here, as we apply activation later
                    qkv_bias = qkv_bias,
            )
            sequence_len_vision = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)])
            if use_pretrained_vit and sequence_len_vision != 196:
                print("Pretrained ViT models are trained with a patch size of 16x16 at image size 224x224: sequence_len_vision must be 196, but is {}".format(sequence_len_vision))
            if use_pretrained_vit: 
                print("Pretrained ViT not yet implemented. Training from scratch.")

            if act == "tanh":
                self.vision_model = nn.Sequential(vision_model, ArgPass(), nn.Dropout(dropout_rate), nn.Tanh())
            elif act == "relu":
                self.vision_model = nn.Sequential(vision_model, ArgPass(), nn.Dropout(dropout_rate), nn.ReLU())
            else:
                self.vision_model = nn.Sequential(vision_model, ArgPass(), nn.Dropout(dropout_rate))


        #CLINICAL
        if self.only_clinical or self.multimodal:
            clinical_mlp_dim = 4*num_clinical_features if not apply_tabular_embedding else 4*tabular_embedding_size #times 4 is quite common
            if apply_tabular_embedding:
                TabularPreprocessing = TabularEmbedding(num_embeddings_tabular, tabular_embedding_size)
                if act == "tanh":
                    self.clinical_model = nn.Sequential(TabularPreprocessing, MLPBlock(tabular_embedding_size, clinical_mlp_dim, dropout_rate), nn.Tanh())
                elif act == "relu":
                    self.clinical_model = nn.Sequential(TabularPreprocessing, MLPBlock(tabular_embedding_size, clinical_mlp_dim, dropout_rate), nn.ReLU())
                else:
                    self.clinical_model = nn.Sequential(TabularPreprocessing, MLPBlock(tabular_embedding_size, clinical_mlp_dim, dropout_rate))
            else:
                if act == "tanh":
                    self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.Tanh())
                elif act == "relu":
                    self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.ReLU())
                else:
                    self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate))
        #update num_clinical_features to the new value
        num_clinical_features = tabular_embedding_size if apply_tabular_embedding else num_clinical_features

        #FUSION
        if self.multimodal:
            fusion_dim = num_clinical_features + num_vision_features #for concatenation of vision and clinical features
            fusion_mlp_dim = 4*fusion_dim
            self.fusion = MLPBlock(fusion_dim, fusion_mlp_dim, dropout_rate)
            # no further act needed here, as Classifier does Sigmoid anyway

        if self.multimodal:
            self.classification_head = Classifier(fusion_dim, num_classes)
        elif self.only_vision:
            self.classification_head = Classifier(num_vision_features, num_classes)
        elif self.only_clinical:
            self.classification_head = Classifier(num_clinical_features, num_classes)

    def forward(self, clinical_info, img):
        if self.multimodal:
            #print("img.shape", img.shape)
            #print("img", img)
            #print("clinical_info", clinical_info)
            #print("clinical_info.shape", clinical_info.shape)
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