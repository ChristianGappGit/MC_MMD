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
    model_name: str
    in_channels: int
    patch_size: Union[Sequence[int], int]
    num_classes: int
    num_vision_layers: int
    num_text_layers: int
    num_cross_attention_layers: int
    num_pre_activation_layers: int
    num_pre_activation_layers_cross: int
    num_attention_heads_text: int
    num_attention_heads_vision: int
    spatial_dims: int
    drop_out: float
    text_only: bool
    vision_only: bool
    serial_pipeline: bool
    llama_path_server: str
    llama_path_local: str
    language_model: str
    tokenizer: str
    vocab_size: int
    dim: int
    multiple_of: int
    vit_path_server: str
    hidden_size_vision: int
    intermediate_size_vision: int
    conv1_t_size: int
    conv1_t_stride: int
    pretrained_vision_net: bool

@dataclass(frozen=True)
class OcclusionParameters:
    overlap: float
    mode: Union[str, float]
    n_batch: int
    text_mask: str
    num_examples: int

@dataclass(frozen=True)
class LoraConfig:
    use_lora: bool
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    target_modules_req_grad: List[str]

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
    lora: LoraConfig
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