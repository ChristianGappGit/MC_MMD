"""
Multimodal Tranformer for vision and text.
"""

import math
import os
import sys
from typing import Tuple, Union, Sequence
import numpy as np
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

import time
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch.distributed as dist

from llama import (
    ModelArgs, 
    Tokenizer, 
    #TransformerBlock as ... #not needed any more (implemented in this file)
)

from vit_monai import ViT   #adapted version of monai's ViT (added "lora_" prefix to nn.Parameter layer (for position embeddings)in PatchEmbeddingBlock)
from monai.networks.layers import trunc_normal_ #used for text_position_embeddings initialization

#from llama.model import RMSNormHP as RMSNormLLaMA #use when memory problems
from llama.model import RMSNorm as RMSNormLLaMA
from llama.model import precompute_freqs_cis
from llama.model import Attention as AttentionLLaMA
from llama.model import FeedForward as FeedForwardLLaMA
from llama.model import apply_rotary_emb

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding, #TODO: check if works well with LLaMA
    ColumnParallelLinear,
    #need layer here...?
)

from monai.utils import ensure_tuple_rep

#LLaMAEmbeddings = ParallelEmbedding #no further var needed, use ParallelEmbedding from fairscale directly instead.

__all__ = ( "Transformer_LLaMA", "LLaMAPretrainedModel", 
           "AttentionLoRAText", "AttentionLoRA", 
            "FeedForwardLoRA_text","FeedForwardLoRA",
            "TransformerBlock_LLaMA", "CrossAttention",
            "LLaMAOutput", "CrossAttentionBlock",
            "DenseLinear", "Classifier", "TextT",
            "MultiModalTransformer",
            "ImageTextNet",
            )

class LLaMAPretrainedModel(nn.Module):
    """
    Large Language Model Meta AI (LLaMA) Pretrained Model
    """
    def __init__(self, model_args, *inputs, **kwargs) -> None:
        assert model_args is not None
        super().__init__()

    def init_LLaMA_weights(self, module):
        if isinstance(module, (nn.Linear, ParallelEmbedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def  from_pretrained(
        cls,    #class self
        llama_config,
        *inputs,
        **kwargs,
    ):
        """
        Setup functions to load LLaMA weights and bias (*.pth) file from a pre-trained model.
        """
        def setup_model_parallel() -> Tuple[int, int]:
            print("Setting up model parallel...")
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            world_size = int(os.environ.get("WORLD_SIZE", -1))

            print(f"Local rank: {local_rank}")
            print(f"World size: {world_size}")

            dist.init_process_group("nccl")# rank=local_rank, world_size=world_size)#, store=dist.FileStore("fiel_store")) #:FIXME: this is the line that fails, endless loop in _rendezvous_helper
            print("Initialized distributed process group")
            initialize_model_parallel(world_size)
            print("Initialized model parallel")
            torch.cuda.set_device(local_rank)

            # seed must be the same in all processes
            torch.manual_seed(1)
            return local_rank, world_size
    
        def load(
            ckpt_dir: str,
            tokenizer_path: str,
            local_rank: int,
            world_size: int,
            max_seq_len: int,
            max_batch_size: int,
        ):
            start_time = time.time()
            checkpoints = sorted(Path(ckpt_dir).glob("*.pth")) #here we could/should load a model pretrained for clinical context
            assert world_size == len(
                checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}" #TODO: That is why models > 7B parameters cannot be loaded on a single GPU
            ckpt_path = checkpoints[local_rank]
            print("Loading")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            #with open(Path(ckpt_dir) / "params.json", "r") as f: #not needed any more
            #    params = json.loads(f.read())
            params = {"dim": llama_config["embedding_dim"], 
                      "multiple_of": llama_config["multiple_of"],
                      "n_heads": llama_config["num_attention_heads"], 
                      "n_layers": llama_config["n_layers"], #irrelevant, could be removed...?!
                      "norm_eps": llama_config["norm_eps"],
                      "vocab_size": llama_config["num_embeddings"],
            }

            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
            )
            tokenizer = Tokenizer(model_path=tokenizer_path)
            model_args.vocab_size = tokenizer.n_words
            llama_config["num_embeddings"] = tokenizer.n_words  #here the vocab_size must be set correctly for the ParallelEmbedding class ("out_features")
            #torch.set_default_tensor_type(torch.cuda.HalfTensor)

            #Following line creates object of MultiModalTransformer (where also Transformer_LLaMA is initialized with model_args)
            model = cls(llama_config, model_args, *inputs, **kwargs)
            #torch.set_default_tensor_type(torch.FloatTensor) #FIXME: maybe delete?
            model.load_state_dict(checkpoint, strict=False)

            """
            Note that the generator is not needed in this case as we don't want to generate text but only ouput probabilities (logits).
            generator = LLaMA(model, tokenizer)
            """
            
            print(f"Loaded in {time.time() - start_time:.2f} seconds")
            print("pretrained llama model tree:")
            print(model)
            print("-------------------------------------------------------")
            return model

        #Model / Tokenizer Paths:
        LLaMA_7 = os.path.join(llama_config["model_path"], "7B")
        LLaMA_13 = os.path.join(llama_config["model_path"], "13B")
        LLaMA_30 = os.path.join(llama_config["model_path"], "30B")
        tokenizer_LLaMA = os.path.join(llama_config["model_path"], "tokenizer.model")

        #Setup model parallel:
        local_rank, world_size = setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        params = {  "local_rank" : local_rank,
                    "world_size" : world_size,
                    "max_seq_len" : llama_config["max_seq_len"],
                    "max_batch_size" : llama_config["max_batch_size"]
        }

        #Load model:
        if llama_config["model_type"] == "LLaMA7B": ckpt_dir_llama = LLaMA_7
        elif llama_config["model_type"] == "LLaMA13B": ckpt_dir_llama = LLaMA_13
        elif llama_config["model_type"] == "LLaMA30B": ckpt_dir_llama = LLaMA_30
        else : raise ValueError("Invalid model_type. Must be one of: LLaMA7B, LLaMA13B, LLaMA30B")

        model = load(ckpt_dir=ckpt_dir_llama, tokenizer_path=tokenizer_LLaMA, **params)
        return model

class AttentionLoRAText(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads # // ("int division") fs_init.get_model_parallel_world_size() #TODO: no multiple GPU support yet
        self.head_dim = args.dim // args.n_heads

        self.wq_text= nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wk_text = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wv_text = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wo_text = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq_text(x), self.wk_text(x), self.wv_text(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            #print warning
            #print("Warning: apply_rotary_emb() was called in AttentionLoRA.forward() method.") #TODO: .... to be deleted, FIXME: check if necessary

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo_text(output)

class AttentionLoRA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads # // ("int division") fs_init.get_model_parallel_world_size() #TODO: no multiple GPU support yet
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
        )
        #deleted caches as not needed for classification but only text generation

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            #print warning
            #print("Warning: apply_rotary_emb() was called in AttentionLoRA.forward() method.") #TODO: .... to be deleted, FIXME: check if necessary

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class FeedForwardLoRAText(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1_text = nn.Linear(
            dim, hidden_dim,
        )
        self.w2_text = nn.Linear(
            hidden_dim, dim,       #TODO: check if this makes sense
        )
        self.w3_text = nn.Linear(
            dim, hidden_dim, 
        )

    def forward(self, x):
        return self.w2_text(F.silu(self.w1_text(x)) * self.w3_text(x))

class FeedForwardLoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim,       #TODO: check if this makes sense
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, 
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock_LLaMA(nn.Module):           #adapted from llama/model.py
    def __init__(self, layer_id: int, args: ModelArgs, modality: str=None): #modality defines layer names in AttentionLoRA
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        if modality == "text":
            self.attention = AttentionLoRAText(args)
            self.feed_forward = FeedForwardLoRAText(
                dim=args.dim, hidden_dim= 4 * args.dim, multiple_of=args.multiple_of,
            )
        else:
            self.attention = AttentionLoRA(args)
            self.feed_forward = FeedForwardLoRA(
                dim=args.dim, hidden_dim= 4 * args.dim, multiple_of=args.multiple_of,
            )
        self.layer_id = layer_id
        self.attention_norm = RMSNormLLaMA(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNormLLaMA(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class CrossAttention(nn.Module):
    """LLaMA Attention layer.
    Based on: LLaMA
    """
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_local_heads = n_heads #// fs_init.get_model_parallel_world_size() #FIXME: does not work for multiple GPU yet
        self.head_dim = dim // n_heads

        self.wq_cross = nn.Linear(
            dim,
            self.n_local_heads * self.head_dim,
        )
        self.wk_cross = nn.Linear(
            dim,
            self.n_local_heads * self.head_dim,
        )
        self.wv_cross = nn.Linear(
            dim,
            self.n_local_heads * self.head_dim,
        )
        self.wo_cross = nn.Linear(  #TODO: check if correct
            self.n_local_heads * self.head_dim,
            dim,
        )
        #DONE: deleted cache as only needed for text generation

    def forward(self, context, hidden_states, start_pos: int, freqs_cis=None, mask=None): #TODO: check and modify if necessary
        """modified version of llama/model.py Attention.forward() method"""
        bsz, seqlen, _ = context.shape
        bsz_q, seqlen_q, _ = hidden_states.shape
        xq, xk, xv = self.wq_cross(hidden_states), self.wk_cross(context), self.wv_cross(context) #query swapped ! (#cross_attention)

        xq = xq.view(bsz_q, seqlen_q, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            #print warning
            #print("Warning: apply_rotary_emb() was called in CrossAttention.forward() method.")

        keys = xk
        values = xv #TODO: check if copy needed? should be ok...

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        #print("Attention(): output_shape_before transpose", output.shape)    #TODO: delete
        output = output.transpose(1, 2)
        bsz_out, seqlen_out, _, _  = output.shape
        output = output.contiguous().view(bsz_out, seqlen_out, -1)  #corrected !

        return self.wo_cross(output)
        

class LLaMAOutput(FeedForwardLoRA): #TODO: could also use FeedForwardLoRA here.
    """LLaMA output layer.
    Based on: LLaMA
    """
    def __init__(self, args: dict) -> None:
        super().__init__(dim=args["dim"], 
                         hidden_dim=4*args["dim"],   #TODO: check: #Transformer MLP feedforward dimension d_ffn = 4 x d_model .
                         multiple_of=args["multiple_of"]
                        )
        self.ffn_norm = RMSNormLLaMA(args["dim"], eps=args["norm_eps"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states)) #TODO: check if correct
        return self.ffn_norm(hidden_states + input_tensor) #TODO: check if needed
        #return (hidden_states+input_tensor)
    
class CrossAttentionBlock(nn.Module):
    """LLaMA cross attention layer.
    Based on: LLaMA
    CONVENTION: x: text, y: vision
    """
    def __init__(self, args_text: dict, args_vision: dict) -> None:
        super().__init__()
        self.attention_x = CrossAttention(dim=args_text["dim"], n_heads=args_text["n_heads"])
        self.attention_y = CrossAttention(dim=args_vision["dim"], n_heads=args_vision["n_heads"]) #new_dim used here
        self.attention_norm = RMSNormLLaMA(args_text["dim"], eps=args_text["norm_eps"]) #same as with vision_args
        self.output_x = LLaMAOutput(args_text)
        self.output_y = LLaMAOutput(args_vision)

    def forward(self, x, y, start_pos: int, freqs_cis=None, mask=None):
        #print("x_shape", x.shape) #TODO: delete
        #print("y_shape", y.shape) #TODO: delete
        out_x = self.attention_x.forward(self.attention_norm(x), self.attention_norm(y), start_pos=start_pos , freqs_cis=freqs_cis, mask=mask)
        out_y = self.attention_y.forward(self.attention_norm(y), self.attention_norm(x), start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        return self.output_x(out_x, x), self.output_y(out_y, y)
    
class DenseLinear(nn.Module):
    """
    DenseLinear layer.
    with tanh activation function.
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        output = self.dense(first_token_tensor)
        output = self.activation(output)
        return output
    
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

class AdaptiveAvgPoolOutput(nn.Module):
    """
    transform the output of text or vision from [b], [s], [d] to [b], [d] using AdaptiveAvgPool1d.
    """

    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        return x
    
    
def load_pretrained_ViT(vit_config: dict):
    """Vision Transformer.
    Based on: Vision Transformer (ViT)
    (Embedding done with PatchEmbeddingBlock from MONAI in VisionTransformer class)
    """
    #init model
    model = ViT(    #imported from MONAI (TODO: add adapted version, in order to add "lora_" prefix to nn.Parameter layer (for position embeddings)
        in_channels=vit_config["in_channels"],
        img_size=vit_config["img_size"],
        patch_size=vit_config["patch_size"],
        hidden_size= vit_config["hidden_size"],
        mlp_dim=vit_config["mlp_dim"],
        num_layers=vit_config["num_layers"],
        num_heads=vit_config["num_heads"],
        pos_embed=vit_config["pos_embed"],
        classification=vit_config["classification"],
        num_classes=vit_config["num_classes"],
        dropout_rate=vit_config["dropout_rate"],
        spatial_dims=vit_config["spatial_dims"],
        post_activation=vit_config["post_activation"],
        qkv_bias=vit_config["qkv_bias"],
    )
    """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block
    """

    if vit_config["pretrained"]:
        #load pretrained weights
        # MODEL 0: 224 x 224
        print('Loading Weights from the Path {}'.format(vit_config["model_path"]))
        vit_weights = torch.load(vit_config["model_path"]) #the loaded model only contains the weights here.. hence torch.load() returns the weights
        # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
        # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
        # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
        # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
        model_dict = model.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.load_state_dict(model_dict)
        del model_dict, vit_weights
        print('Pretrained Weights from ImageNet1k Succesfully Loaded !')

        # MODEL 1: 256 x 256
        # not implemented yet
    else:
        print('Training from Scratch. No weights loaded.')

    return model

class TextT(LLaMAPretrainedModel):
    """Text Transformer.
    Based on: LLaMA
    """
    def __init__(self, llama_config, model_args, *inputs, **kwargs) -> None:

        super().__init__(model_args)
        
        self.config = type("obj", (object,), llama_config)
        assert self.config.num_embeddings != -1 #must have been set correctly by tokenizer in LLaMAPretrainedModel by "super().__init__()"
        #self.text_embeddings = ParallelEmbedding(num_embeddings=self.config.num_embeddings, 
        #                                    embedding_dim=self.config.embedding_dim, init_method=lambda x: x)
        
        self.text_embeddings = torch.nn.Embedding(num_embeddings=self.config.num_embeddings, embedding_dim=self.config.embedding_dim) #TODO: check if works properly
        self.lora_text_position_embeddings = nn.Parameter(torch.zeros(1, self.config.max_seq_len, self.config.embedding_dim))# num_embeddings = vocab_size, embedding_dim = dim
        trunc_normal_(self.lora_text_position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0) #initialization of position embeddings
        #NOTE: no multiple GPU usage yet when using nn.Embedding
        #TODO: model_args: check if updated correctly
        self.text_encoder = nn.ModuleList([TransformerBlock_LLaMA(layer, model_args,"text") for layer in range(llama_config["n_layers"])])
        self.apply(self.init_LLaMA_weights)

    def forward(self, input_ids, start_pos, freqs_cis=None, mask=None): #set default values to None?
        #print("text_embeddings.shape", self.text_embeddings.weight.shape) #TODO: delete
        #print("lora_text_position_embeddings.shape", self.lora_text_position_embeddings.shape) #TODO: delete
        text_features = self.text_embeddings(input_ids,) #token_type_ids (buffer) not used here
        text_features = text_features + self.lora_text_position_embeddings
        #_bsz, seqlen = input_ids.shape #not needed here anyway
        embedded_features = text_features.clone()
        for layer in self.text_encoder:
            text_features = layer(text_features, start_pos, freqs_cis, mask)
        return text_features, embedded_features

class MultiModalTransformer(nn.Module):
    """
    Multimodal Tranformer with 
    LLaMA and ViT backbones.
    """
    
    def __init__(
        self , cross_config: dict, llama_config: dict, vit_config: dict, controller_config: dict, num_pre_activation_layers: int,
    ) -> None:
        print("------------------------------------------------------")
        """
        Args:
            num_cross_attention_layers: number of layers used for mutual information (cross attention).
            llama_config: configuration for LLaMA transformer
            model_args: model arguments for LLaMA transformer
            vit_config: configuration for Vision Transformer
            controller_config: configuration for controller
        """
        super().__init__()
        self.serial_pipeline = controller_config["serial_pipeline"]
        self.text_only = controller_config["text_only"]
        self.vision_only = controller_config["vision_only"]
        
        if not self.vision_only :   self.text_net = TextT.from_pretrained(llama_config)
        if not self.text_only :     self.vision_net = load_pretrained_ViT(vit_config)

        model_args_text = {  "dim": llama_config["embedding_dim"],
                    "multiple_of": llama_config["multiple_of"],
                    "n_heads": llama_config["num_attention_heads"],
                    "n_layers": llama_config["n_layers"],
                    "norm_eps": llama_config["norm_eps"],
                    "vocab_size": llama_config["num_embeddings"],
                    "max_seq_len": llama_config["max_seq_len"],
                    "max_batch_size": llama_config["max_batch_size"],
        }

        vit_config["num_embeddings"] = llama_config["num_embeddings"] #for output adjustment

        model_args_vision = {  "dim": model_args_text["dim"],
                    "old_dim": vit_config["hidden_size"],              #!!!!!!!!
                    "multiple_of": vit_config["multiple_of"],
                    "n_heads": vit_config["num_heads"],
                    "n_layers": vit_config["num_layers"],
                    "norm_eps": vit_config["norm_eps"],
                    "mlp_dim": vit_config["mlp_dim"],
                    "vocab_size":  vit_config["num_embeddings"], 
        }

        self.transform_vision = nn.Linear(model_args_vision["old_dim"], model_args_vision["dim"]) #TODO: check if correct
        self.norm_text = RMSNormLLaMA(model_args_text["dim"], eps=model_args_text["norm_eps"])
        self.norm_vision = RMSNormLLaMA(model_args_vision["dim"], eps=model_args_vision["norm_eps"])

        #self.output = ColumnParallelLinear(
        #    model_args.dim, model_args.vocab_size, bias=False, init_method=lambda x: x
        #)
        self.avg_pool_text = AdaptiveAvgPoolOutput()
        self.avg_pool_vision = AdaptiveAvgPoolOutput()
        self.freqs_cis = precompute_freqs_cis(
            model_args_text["dim"] // model_args_text["n_heads"], model_args_text["max_seq_len"] * 2
        )

        #TODO: check for pre_activation_encoder... dimensions.
        self.cross_attention_encoder = nn.ModuleList([CrossAttentionBlock(model_args_text, model_args_vision,) for _ in range(cross_config["num_cross_attention_layers"])])
        #self.pre_activation_encoder = nn.ModuleList([TransformerBlock_LLaMA(layer, model_args, "") for layer in range(num_pre_activation_layers)]) #deactivated for now.
        self.pre_activation_cross_encoder = nn.ModuleList([CrossAttentionBlock(model_args_text, model_args_vision,) for _ in range(cross_config["num_pre_activation_layers_cross"])])

    def forward(self, input_ids, start_pos: int, token_type_ids=None, image=None, attention_mask=None,
                vision_available: bool = False, text_available: bool = False):

        #if vision_features is not None and vision_features.type() is not torch.cuda.HalfTensor:
        #    vision_features = vision_features.half()
        #print("vision_features_type was changed from torch.cuda.FloatTensor to", vision_features.type())
            
        freqs_cis_text = None #TODO: changed to None as not needed here!
        freqs_cis_vision = None #TODO: changed to None as not needed here!

        """ # see also Transformer forward method for further information
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        """
        
        #---------------------------------------------------------------------------------------------------------------
        #TEXT ONLY
        if self.text_only or (text_available and not vision_available):
            #check for num_layers > 0 removed. BEWARE
            text_features, _ = self.text_net(input_ids, start_pos, freqs_cis=freqs_cis_text, mask=attention_mask)
            text_features = self.norm_text(text_features)
            output_text = self.avg_pool_text(text_features)
            output_vision = None
            return output_vision, output_text
        #---------------------------------------------------------------------------------------------------------------
        #VISION ONLY
        elif self.vision_only or (vision_available and not text_available):
            #check for num_layers > 0 removed. BEWARE
            vision_features, _, _ = self.vision_net(image)
            vision_features = self.transform_vision(vision_features) #transform hidden_size_vision to dim (text_features)
            vision_features = self.norm_vision(vision_features)
            output_vision = self.avg_pool_vision(vision_features)  # only compute last logits
            output_text = None
            return output_vision, output_text
        #---------------------------------------------------------------------------------------------------------------
        #MULTIMODAL
        elif vision_available and text_available: #Multimodality
            
            text_features, embedded_text_features = self.text_net(input_ids, start_pos, freqs_cis=freqs_cis_text, mask=attention_mask)
            vision_features, ebedded_vision_features, _ = self.vision_net(image)

            #print("text_features.shape", text_features.shape) #TODO: delete
            #print("vision_features.shape", vision_features.shape) #TODO: delete #(batch_size, seq_len, num_patches=hidden_size)
            #vision_features: (batch_size, seq_len, hidden_size)
            #text_features: (batch_size, seq_len, dim)
            #batch_size, text_seq_len, dim = text_features.shape
            #_, vision_seq_len, num_patches = vision_features.shape

            #adapt seq_len of vision_features to seq_len of text_features
            #vision_features = vision_features.view(batch_size, text_seq_len, -1) #TODO: check if correct
            vision_features = self.transform_vision(vision_features) #transform hidden_size_vision to dim (text_features)
            #print("vision_features.shape", vision_features.shape) #TODO: delete
            
            #------------------------------------------CROSS LAYER----------------------------------------------------
            if len(self.cross_attention_encoder) > 0:
                if self.serial_pipeline:
                    for layer in self.cross_attention_encoder:
                        text_features, vision_features = layer(text_features, vision_features, start_pos, freqs_cis=None, mask=None) #FIXME: check freqs_cis
                else: #parallel_pipeline
                    #init cross_features
                    vision_features_cross = self.transform_vision(ebedded_vision_features) #beware that *pre_features will be changed herein afterwards
                    text_features_cross = embedded_text_features
                    for layer in self.cross_attention_encoder:
                        text_features_cross, vision_features_cross = layer(text_features_cross, vision_features_cross, start_pos, freqs_cis=None, mask=None) #FIXME: check freqs_cis
                    #summation
                    vision_features = vision_features + vision_features_cross
                    text_features = text_features + text_features_cross
            #--------------------------------------------------------------------------------------------------------

            #for layer in self.pre_activation_encoder: #Concatenation (note that one layer is used for both modalities, no cross attention)
            #    text_features = layer(text_features, start_pos, freqs_cis=freqs_cis_text, mask=None) #TODO: mask = ?
            #    vision_features = layer(vision_features, start_pos, freqs_cis=freqs_cis_vision, mask=None)
            
            for layer in self.pre_activation_cross_encoder:
                text_features, vision_features = layer(text_features, vision_features, start_pos, freqs_cis=None, mask=None)
            
            vision_features = self.norm_vision(vision_features)
            output_vision = self.avg_pool_vision(vision_features)  # only compute last logits
            text_features = self.norm_text(text_features)
            output_text = self.avg_pool_text(text_features)

            return output_vision, output_text

        else: #no modality available
            raise ValueError("No modality available.")


class ImageTextNet(torch.nn.Module):
    """
    ImageTextNet is a multimodal Transformer for Vision and Language.
    The architecture contains a Vision Transformer (ViT) for image input and a BERT/LLaMA for text.
    Both transformers are build of defineable number of layers. Cross attention layers for 
    mutual information between the two modalidies is directly placed after the two last layers.
    The number of Cross attention layers is also of variable size.
    Image text pairs are analysed within the model -> a classification is done.

    Handling of incomplete data:
    Incomplete image text pairs can be handled too.
    Herein the output of both the cross attention layer and the other modalidy's last layer are None.
    ...

    """

    def  __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        num_classes: int,
        num_vision_layers: int,
        num_text_layers: int,
        num_cross_attention_layers: int,
        num_pre_activation_layers: int, #immediately before activation
        num_pre_activation_layers_cross: int, # --||-- (with cross attention)
        llama_path: str, 
        vit_path: str, 
        spatial_dims: int = 3,
        hidden_size_vision: int = 768,
        drop_out: float = 0.0,
        text_only: bool = False,
        vision_only: bool = False,
        serial_pipeline: bool = True, #added
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        intermediate_size_vision: int = 3072,
        language_model: str = "LLaMA7B", #possible options: LLaMA7B, LLaMA13B, LLaMA30B, bert
        num_attention_heads_text: int = 12, 
        num_attention_heads_vision: int = 12,
        pad_token_id: int = 0,
        text_max_seq_len: int = 128, #matches length of sequence_lenght
        vocab_size: int = 32768,
        dim: int = 4096,  #added, used for embedding_dim in Embedding class
        multiple_of = 256, #added, used for multiple_of in TransformerBlock class
    ) -> None:
        """
        Args:
            in_channels: number of input channels (regarding the images)
            img_size: dimension of input image.
            patch_size: dimension of patch_size.
            num_classes: number of classer dor classification.
            num_vision_layers: number of vision transformer layers.
            num_text_layers: number of text transformer layers.
            num_cross_attention_layers: number of layer used for mutual information (cross attention).
            num_pre_activation_layers: number of layer used before activation (no cross attention)
            num_pre_activation_layers_cross: number of layer used before activation (with cross attention)
            spatial_dims: dimension of spatial size.
            hidden_size_vision: typically 768 (default value used in monai ViT)
            drop_out: fraction of the input units to drop,
            text_only: if True, only the text modality is used.
            vision_only: if True, only the vision modality is used.
            serial_pipeline: if True, the cross attention layers are applied after the text- and vision layers, if False, they are applied next to them (using the llama's output).
        
        The other parameters are part of the 'bert_config' to 'MultiModal.from_pretrained'.
        """
        
        super().__init__()

        cross_config = {
            "num_cross_attention_layers" : num_cross_attention_layers,
            "num_pre_activation_layers_cross" : num_pre_activation_layers_cross,
        }

        controller_config = { #contains parameters for controlling of llama and vit
            "serial_pipeline": serial_pipeline,
            "text_only": text_only,
            "vision_only": vision_only,
        }

        assert num_attention_heads_text == num_attention_heads_vision, "num_attention_heads_text and num_attention_heads_vision must be equal."

        llama_config = {
            "model_path": llama_path,
            "model_type": language_model,
            "num_embeddings": vocab_size,
            "embedding_dim": dim,
            "multiple_of": multiple_of,
            "num_attention_heads": num_attention_heads_text,
            "n_layers": num_text_layers,
            "norm_eps": 1e-6,
            "max_seq_len": text_max_seq_len,
            "max_batch_size": 32,
            "hidden_size": dim,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "initializer_range": initializer_range,
        }

        vit_config = {
            "model_path": vit_path,
            "in_channels": in_channels,
            "img_size": img_size,
            "patch_size": patch_size,
            "hidden_size": hidden_size_vision,
            "mlp_dim": intermediate_size_vision,    #default value used here (see monai ViT)
            "num_layers": num_vision_layers,
            "num_heads": num_attention_heads_vision,
            "pos_embed": "conv",
            "classification": False,    #not True, as classification is not done at this point
            "num_classes": None, #not relevant here since no classification
            "dropout_rate": hidden_dropout_prob,
            "spatial_dims": spatial_dims,
            "post_activation": None,    #not relevant since no classification
            "qkv_bias": False,
            "pretrained": True,
            "norm_eps": 1e-6,
            "multiple_of": multiple_of,
            "num_embeddings": None, #set later
            "vision_seq_len": np.prod(img_size) // np.prod(patch_size),
        }

        if text_max_seq_len != vit_config["vision_seq_len"]:
            raise ValueError("text_max_seq_len (= %d) and vision_seq_len (= %d) must be equal." \
                            % text_max_seq_len, vit_config["vision_seq_len"])
        
        if text_only and vision_only:
            raise ValueError("text_only and vision_only cannot be True at the same time.")
        self.text_only = text_only #for foward() method
        self.vision_only = vision_only

        if not (0 <= drop_out <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if (m < p):
                raise ValueError("patch_size should be smaller than img_size.")

            if (m % p != 0):  # type: ignore
                raise ValueError("img_size should be divisible by patch_size.")

        if language_model == "bert":
            NotImplementedError("Bert removed. Use LLaMA instead.")

        elif language_model == "LLaMA7B" or language_model == "LLaMA13B" or language_model == "LLaMA30B":
            self.multimodal = MultiModalTransformer(
                cross_config=cross_config,
                controller_config=controller_config,
                llama_config=llama_config,
                vit_config=vit_config,
                num_pre_activation_layers=num_pre_activation_layers,
            )
            #torch.set_default_tensor_type(torch.cuda.HalfTensor)
            intermediate = dim // 2
            self.dense = DenseLinear(dim, intermediate) #TODO: check !!!
            self.cls_head = Classifier(intermediate, num_classes)
            #torch.set_default_tensor_type(torch.FloatTensor)
        else:
            raise ValueError("language_model should be either 'LLaMA7B' or 'LLaMA13B'.")

        self.drop = torch.nn.Dropout(drop_out)
        
    def forward(self, input_ids, image=None, token_type_ids=None):   ##__call__() #token_type_ids not used for llama
        #token_type_ids ... buffer
        start_pos = 0 #TODO: can always be zero as all tokens are pushed through the model at once, since no text generation is done
        use_vision = (image is not None)

        #Text Embedding done in TextT
        #Vision Embedding done in ViT
        
        use_text = (input_ids is not None)
        if use_text:
            attention_mask = torch.ones_like(input_ids).unsqueeze(1).unsqueeze(2) #reshape tensor to dimension of size 1 at specific position
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        if not use_vision and not use_text:
            raise ValueError("Both modality features are None. At least one of the modalities must be used.")
        hidden_state_vision, hidden_state_text = self.multimodal(
            input_ids=input_ids, start_pos=start_pos, token_type_ids=token_type_ids, image=image, attention_mask=attention_mask,
            vision_available=use_vision, text_available=use_text,
        )
        #pool multimodal output: dense + activation (tanH)
        pooled_text_features = self.dense(hidden_state_text) if hidden_state_text is not None else None
        pooled_vision_features = self.dense(hidden_state_vision) if hidden_state_vision is not None else None
        #features have same shape now (and can be added together)

        #print("pooled_text_features_shape", pooled_text_features.shape) #TODO: delete
        #print("pooled_vision_features_shape", pooled_vision_features.shape)

        if (use_text and not use_vision) or self.text_only:
            logits = self.drop(pooled_text_features)
        elif (use_vision and not use_text) or self.vision_only:
            logits = self.drop(pooled_vision_features)
        elif use_text and use_vision:
            logits = self.drop(pooled_text_features + pooled_vision_features) #addition of pooled features and dropout
        else: #both False #already checked above
            raise ValueError("At least one of the modalities must be used.")
        
        #dense (to num_classes) + activation (Sigmoid)
        logits = self.cls_head(logits)
        #print("logits_shape", logits.shape) #TODO: delete
        return logits