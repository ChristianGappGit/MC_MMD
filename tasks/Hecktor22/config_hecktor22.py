import yaml
from typing import Optional, Tuple, List, Union, Sequence, TYPE_CHECKING
import dataclasses
from copy import copy


# this is only for type checking with vscode and python-language-server
# see https://github.com/microsoft/python-language-server/issues/1898
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class NetParameters:
    arch: str
    in_channels: int
    img_size: Union[Sequence[int], int]
    patch_size: Union[Sequence[int], int]
    spatial_dims: int
    num_classes: int
    num_clinical_features: int
    hidden_size: int
    mlp_dim: int
    num_heads: int
    num_vision_layers: int
    num_clinical_layers: int
    num_cross_layers: int
    dropout_rate: float
    qkv_bias: bool
    act_ViT: str
    conv1_t_size: int
    conv1_t_stride: int
    pretrained_vision_net: bool
    model_path: Optional[str]
    act: str
    only_vision: bool
    only_clinical: bool

@dataclass(frozen=True)
class TrainParameters:
    epochs: int
    batch_size: int
    randFlipProp: float
    randAffineProp: float
    randAffineRotateRange: Tuple[float, float, float]
    randAffineScaleRange: Tuple[float, float, float]

@dataclass(frozen=True)
class ValParameters:
    batch_size: int
    interval: int

@dataclass(frozen=True)
class DataParameters:
    use_segmented_imgs : bool
    cache: bool
    use_only_CT : bool
    apply_z_norm: bool

@dataclass(frozen=True)
class OptimizerParameters:
    learning_rate: float
    weight_decay: float

@dataclass(frozen=True)
class OcclusionParameters:
    mask_patches: bool
    mask_size: Union[Sequence[int], int]
    overlap: float
    mode: Union[str, float]
    n_batch: int
    tabular_mask: bool
    num_examples: int
    do_3D_occlusion: bool

@dataclass(frozen=True)
class Config:
    experiment_name : str
    device: str
    seed: int
    spacing: Sequence[float]
    data: DataParameters
    train: TrainParameters
    val: ValParameters
    net: NetParameters
    optimizer: OptimizerParameters
    occlusion: OcclusionParameters


def deep_dict_union(dict_left: dict, dict_right: dict) -> dict:
    d = copy(dict_left)
    for k, value_right in dict_right.items():
        value_left = d.get(k)
        if isinstance(value_left, dict) and isinstance(value_right, dict):
            d[k] = deep_dict_union(value_left, value_right)
        else:
            d[k] = value_right
    return d


def load_config(default_path: str, config_paths: List[str]=None) -> Config:
    with open(default_path) as file:
        config = yaml.safe_load(file)
        if config_paths is not None:
            for e in config_paths: #overwrite existing parameters from default file
                with open(e) as config_file:
                    config = deep_dict_union(
                        config,
                        yaml.safe_load(config_file) 
                    )
    return Config(**config)