"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

ResNet + MLP (down transform) + tanh() for Vision
MLP + tanh() for Clinical Data
MLP for Fusion
Cls Head (dense + sigmoid) for Classification

Optional Embedding:
    Use with awareness!
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks.mlp import MLPBlock #created for vision, but usable for clinical (i.e. tabular here) data as well
from monai.networks.nets import ResNet as ResNetMonai
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

class ResMLPNet(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            img_size: Union[Sequence[int], int],
            spatial_dims: int,
            num_classes=1, 
            num_clinical_features=10,
            dropout_rate=0.1,
            apply_tabular_embedding=False,
            tabular_embedding_size=96,
            num_embeddings_tabular = 32000, #default for llamaII
            conv1_t_size=7,
            conv1_t_stride=1,
            pretrained_vision_net = False,
            model_path = None, 
            act = "tanh",
            only_vision = False,
            only_clinical = False
            ):
        super().__init__()
        
        self.only_vision = only_vision
        self.only_clinical = only_clinical
        self.multimodal = not only_vision and not only_clinical

        assert not (only_vision and only_clinical), "Only one of only_vision and only_clinical can be True. Set both to False in order to use multimodal model."

        #VISION
        num_vision_features = num_clinical_features if not apply_tabular_embedding else tabular_embedding_size #set to the same number of features
        if self.only_vision or self.multimodal:
            vision_model = ResNetMonai(
                block = "basic",
                layers = [3, 4, 6, 3],
                block_inplanes = [64, 128, 256, 512],
                spatial_dims = spatial_dims,
                n_input_channels=in_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=False,
                shortcut_type="B",
                widen_factor= 1.0,
                num_classes = num_vision_features, #Attention here !!!
                feed_forward = True,
                bias_downsample = True,  # for backwards compatibility (also see PR #5477)
            )
            if pretrained_vision_net and model_path is not None:
                #load weights from ImageNet into model
                print("Pretrained ResNet not yet implemented. Training from scratch.")
                #now load weights:
                vision_model.load_state_dict(torch.load(model_path))
            else:
                print("Training ResNet from scratch.")
            if act == "tanh":
                self.vision_model = nn.Sequential(vision_model, nn.Dropout(dropout_rate), nn.Tanh())
            elif act == "relu":
                self.vision_model = nn.Sequential(vision_model, nn.Dropout(dropout_rate), nn.ReLU())
            else:
                self.vision_model = nn.Sequential(vision_model, nn.Dropout(dropout_rate))

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
            #print("mixed_features", mixed_features, flush=True)
            x = self.fusion(mixed_features)
            #print("x.shape after fusion", x.shape)
            #print("x_out_fusion", x, flush=True)
            x = self.classification_head(x)
            #print("x.shape after classification", x.shape)
            #print("x_output", x, flush=True)
            return x
        elif self.only_vision:
            vision_features = self.vision_model(img)
            x = self.classification_head(vision_features)
            return x
        elif self.only_clinical:
            tabular_features = self.clinical_model(clinical_info)
            x = self.classification_head(tabular_features)
            return x