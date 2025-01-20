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


@dataclass(frozen=True) # If frozen is true, fields may not be assigned to after instance creation.
class DataLoaderParameters:
    batch_size: int
    num_workers: int
    shuffle: bool

@dataclass(frozen=True)
class TrainParameters:
    num_items: int
    dataload: DataLoaderParameters

@dataclass(frozen=True)
class OptimizerParameters:
  learning_rate: float
  weight_decay: float

@dataclass(frozen=True)
class NetConfig:
    name: str
    in_channels: int
    num_classes: int
    #num_clinical_features: int
    spatial_dims: int
    drop_out: float
    apply_tabular_embedding: bool
    tabular_embedding_size: int
    text_only: bool
    vision_only: bool
    llama_path_server: str
    tokenizer: str
    conv1_t_size: int
    conv1_t_stride: int
    pretrained_vision_net: bool
    model_path: Optional[str]
    act: str
    num_vision_layers: int
    num_heads: int
    mlp_dim: int
    hidden_size: int
    patch_size: Union[Sequence[int], int]
    qkv_bias: bool

@dataclass(frozen=True)
class OcclusionParameters:
    overlap: float
    mode: Union[str, float]
    #mask_size: Union[Sequence[int], int] #delteted
    n_batch: int
    text_mask: str
    num_examples: int

@dataclass(frozen=True)
class PerformanceConfig:
    text_available: bool
    vision_available: bool


@dataclass(frozen=True)
class Config:
    device: str
    experiment_name: str
    seed: Optional[int]
    server: bool
    spacing: Sequence[float]
    image_size: Union[Sequence[int], int]
    preprocess_text: bool
    text_max_seq_length: int
    epochs: int
    val_interval: int
    optimizer_param: OptimizerParameters
    train: TrainParameters
    val: TrainParameters
    test: TrainParameters
    net: NetConfig
    occlusion: OcclusionParameters
    performance: PerformanceConfig


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