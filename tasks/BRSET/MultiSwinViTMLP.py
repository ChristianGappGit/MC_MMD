"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

SwinViT(monai) + MLP (down transform) + tanh() for Vision
MLP + tanh() for Clinical Data
MLP for Fusion
Cls Head (dense + sigmoid) for Classification
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks.mlp import MLPBlock #created for vision, but usable for clinical (i.e. tabular here) data as well
from monai.networks.nets.swin_unetr import SwinTransformer
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
    
class SwinViT(SwinTransformer):
    """
    use monai's SwinTransformer, but overwrite forward method (add cls_token)
    """
    def __init__(self,
                in_chans: int,
                embed_dim: int,
                window_size: Sequence[int],
                patch_size: Sequence[int],
                depths: Sequence[int],
                num_heads: Sequence[int],
                mlp_ratio: float = 4.0,
                qkv_bias: bool = True,
                drop_rate: float = 0.0,
                attn_drop_rate: float = 0.0,
                drop_path_rate: float = 0.0,
                norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
                patch_norm: bool = False,
                use_checkpoint: bool = False,
                spatial_dims: int = 3,
                downsample="merging",
                #added for classification
                num_classes: int = 1,
                post_activation: str = "Tanh",
                classification: bool = True,
                **kwargs
                ):
        super().__init__( #FIXME: some error in here.. requires gradient is not True during training (thus no updates)
                in_chans=in_chans,
                embed_dim=embed_dim,
                window_size=window_size,
                patch_size=patch_size,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                patch_norm=patch_norm,
                use_checkpoint=use_checkpoint,
                spatial_dims=spatial_dims,
                downsample=downsample,
        )

        self.classification = classification
        if self.classification:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten(1)
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(embed_dim*2**len(depths), num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(embed_dim*2**len(depths), num_classes)

    #overwrite forward method
    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x) #only impact from spatial_dims is here, from now on irrelevant
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        #print("x4_out.shape", x4_out.shape)
        if self.classification:
            x = self.avgpool(x4_out)
            #print("x.shape after avgpool", x.shape)
            x = self.flatten(x)
            #print("x.shape after flatten", x.shape)
            x = self.classification_head(x)
            #print("x.shape after classification", x.shape)
        else:
            x = [x0_out, x1_out, x2_out, x3_out, x4_out] #Attention what to do with this output in forward() method (not relevant here!!!)
        return x

class SwinViTMLPNet(nn.Module):
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
            apply_tabular_embedding=False,
            tabular_embedding_size=96,
            num_embeddings_tabular = 32000, #default for llamaII
            act = "tanh",
            qkv_bias=False,
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

        #VISION
        num_vision_features = num_clinical_features if not apply_tabular_embedding else tabular_embedding_size #set to the same number of features
        if self.only_vision or self.multimodal:
            vision_model = SwinViT(
                    in_chans=in_channels,
                    embed_dim=embed_dim,
                    window_size=window_size,
                    patch_size=patch_size,
                    depths=depths,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=qkv_bias,
                    drop_rate=dropout_rate,
                    attn_drop_rate=dropout_rate,
                    drop_path_rate=dropout_rate,
                    norm_layer=nn.LayerNorm,
                    patch_norm=False,
                    use_checkpoint=True,
                    spatial_dims=spatial_dims,
                    downsample="merging",
                    num_classes=num_vision_features, #attention here !!!
                    post_activation=None, #handled separately afterwards in "vision_model"
                    classification=True,
                )
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