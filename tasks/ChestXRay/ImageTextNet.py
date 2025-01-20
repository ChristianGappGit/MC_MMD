"""
Multimodal Tranformer for vision and text.
"""

import math
import os
import shutil
import sys
import tarfile
import tempfile
from typing import Tuple, Union, Sequence
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import time
import json
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch.distributed as dist

from llama import (
    ModelArgs, 
    Transformer as Transformer_LLaMA,
    Tokenizer, 
    TransformerBlock as TransformerBlock_LLaMA,
)

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

from monai.utils import optional_import, ensure_tuple_rep
from monai.networks.layers import Conv #Layer Factory
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

transformers = optional_import("transformers")
load_tf_weights_in_bert = optional_import("transformers", name="load_tf_weights_in_bert")
BertEmbeddings = optional_import("transformers.models.bert.modeling_bert", name="BertEmbeddings")[0]
#LLaMAEmbeddings = ParallelEmbedding #no further var needed, use ParallelEmbedding from fairscale directly instead.
cached_path = optional_import("transformers.file_utils", name="cached_path")[0]
BertLayer = optional_import("transformers.models.bert.modeling_bert", name="BertLayer")[0]
#LLaMALayer = ... #used TransformerBlock from "llama" as TransformerBlock_LLaMA instead


__all__ = ("LLaMAPretrainedModel", "BertPreTrainedModel", "CrossAttentionBert", "CrossAttentionLLaMA",
            "BertOutput", "LLaMAOutput", "CrossAttentionBlockLLaMA", "CrossAttentionLayerBert",
            "PoolerBert", "PoolerLLaMA", "MultiModalTransformerLLaMA", "MultiModalTransformerBert" ,"ImageTextNet")


class LLaMAPretrainedModel(Transformer_LLaMA):
    """
    Large Language Model Meta AI (LLaMA) Pretrained Model
    """
    def __init__(self, model_args, *inputs, **kwargs) -> None:
        assert model_args is not None
        super().__init__(model_args)

    def init_LLaMA_weights(self, module):
        if isinstance(module, (nn.Linear, ParallelEmbedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained( #TODO: check what this does...
        cls,    #class self
        num_vision_layers,      #TODO: use gray highlighted vars here...
        num_text_layers,
        num_cross_attention_layers,
        llama_config, #TODO: add vars to llama_config.
        #state_dict=None, #deleted
        #cache_dir=None, #deleted
        #from_tf=False, #deleted
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
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
            ckpt_path = checkpoints[local_rank]
            print("Loading")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            #with open(Path(ckpt_dir) / "params.json", "r") as f: #not needed any more
            #    params = json.loads(f.read())
            params = {"dim": llama_config["embedding_dim"], 
                      "multiple_of": llama_config["multiple_of"],
                      "n_heads": llama_config["num_attention_heads"], 
                      "n_layers": llama_config["n_layers"],
                      "norm_eps": llama_config["norm_eps"],
                      "vocab_size": llama_config["num_embeddings"],
            }

            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
            )
            tokenizer = Tokenizer(model_path=tokenizer_path)
            model_args.vocab_size = tokenizer.n_words
            llama_config["num_embeddings"] = tokenizer.n_words  #here the vocab_size must be set correctly for the ParallelEmbedding class ("out_features")
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

            #Following line creates object of MultiModalTransformerLLaMA (where also Transformer_LLaMA is initialized with model_args)
            model = cls(num_vision_layers, num_text_layers, num_cross_attention_layers, llama_config, model_args, *inputs, **kwargs)
            torch.set_default_tensor_type(torch.FloatTensor) #FIXME: maybe delete?
            model.load_state_dict(checkpoint, strict=False)

            """
            Note that the generator is not needed in this case as we don't want to generate text but only ouput probabilities (logits).
            generator = LLaMA(model, tokenizer)
            """
            
            print(f"Loaded in {time.time() - start_time:.2f} seconds")
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
    

class BertPreTrainedModel(nn.Module):
    """Module to load BERT pre-trained weights.
    Based on:
    LXMERT
    https://github.com/airsplay/lxmert
    BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """
    
    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__()

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,    #class self
        num_vision_layers,
        num_text_layers,
        num_cross_attention_layers,
        bert_config,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        *inputs,
        **kwargs,
    ):
        archive_file = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        model = cls(num_vision_layers, num_text_layers, num_cross_attention_layers, bert_config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu" if not torch.cuda.is_available() else None)
        if tempdir:
            shutil.rmtree(tempdir)
        if from_tf:
            weights_path = os.oath.join(serialization_dir, "model.ckpt")
            return load_tf_weights_in_bert(model, weights_path)
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        
        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(s.startswith("bert.") for s in state_dict.keys()):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        return model


class CrossAttentionBert(nn.Module):
    """BERT attention layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        print("Attention(): new_x_shape", new_x_shape)    #TODO: delete
        x = x.view(*new_x_shape) #reshape
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context):  # query swapped       
        """
        Computation of
        DropOut{ SoftMax [ (Qh * Kc^T) / sqrt(d) ] } * Vc,
        with c..context, h..hidden_states
        """
        print("Attention(): hidden_states_shape", hidden_states.shape)    #TODO: delete
        print("Attention(): context_shape", context.shape)    #TODO: delete
        cross_query_layer = self.query(hidden_states)
        cross_key_layer = self.key(context)
        cross_value_layer = self.value(context)
        print("Attention(): cross_query_layer_shape", cross_query_layer.shape)    #TODO: delete
        print("Attention(): cross_key_layer_shape", cross_key_layer.shape)    #TODO: delete
        print("Attention(): cross_value_layer_shape", cross_value_layer.shape)    #TODO: delete
        query_layer = self.transpose_for_scores(cross_query_layer)
        key_layer = self.transpose_for_scores(cross_key_layer)
        value_layer = self.transpose_for_scores(cross_value_layer)
        print("Attention(): query_layer_shape", query_layer.shape)    #TODO: delete
        print("Attention(): key_layer_shape", key_layer.shape)    #TODO: delete
        print("Attention(): value_layer_shape", value_layer.shape)    #TODO: delete
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #better gradient
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous() #returns a contiguous in memory tensor with same data
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        print("Attention(): context_layer_shape", context_layer.shape)    #TODO: delete
        return context_layer


class BertOutput(nn.Module):
    """BERT output layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12) #per element
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #preventing the co-adaptation of neurons by
                                                              #randomly zeroing some of the elements of the 
                                                              #input thensor

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) #input from layer n added to output of layer n and hence to input of layer n+1
                                                                     #to prevent net from loss of information within a layer
        return hidden_states
    
class CrossAttentionLLaMA(AttentionLLaMA):
    """LLaMA Attention layer.
    Based on: LLaMA
    """
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.num_attention_heads = args.n_heads
        self.attention_head_size = int(args.dim / args.n_heads)

    def forward(self, hidden_states, context, start_pos: int, freqs_cis=None, mask=None): #TODO: check and modify if necessary
        """modified version of llama/model.py Attention.forward() method"""
        print("Attention(): hidden_states_shape", hidden_states.shape)    #TODO: delete
        print("Attention(): context_shape", context.shape)    #TODO: delete
        bsz, seqlen, _ = context.shape
        bsz_q, seqlen_q, _ = hidden_states.shape #FIXME: change to hidden_states.shape again
        xq, xk, xv = self.wq(hidden_states), self.wk(context), self.wv(context) #query swapped ! (#cross_attention)

        xq = xq.view(bsz_q, seqlen_q, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        print("xq_shape",xq.shape) #FIXME: delete
        print("xk_shape",xk.shape) #FIXME: delete
        print("xv_shape",xv.shape) #FIXME: delete

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            #print warning
            print("Warning: apply_rotary_emb() was called in CrossAttentionLLaMA.forward() method.")

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        print("Attention(): output_shape_before transpose", output.shape)    #TODO: delete
        output = output.transpose(1, 2)
        bsz_out, seqlen_out, _, _  = output.shape
        output = output.contiguous().view(bsz_out, seqlen_out, -1)  #corrected !

        print("Attention(): output_shape", output.shape)    #TODO: out_shape must be different !!! FALSE !!!

        return self.wo(output)  #Error here...!!!
        

class LLaMAOutput(FeedForwardLLaMA):
    """LLaMA output layer.
    Based on: LLaMA
    """
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(dim=args.dim, 
                         hidden_dim=4*args.dim,   #TODO: check if correct, see also Transformer class
                         multiple_of=args.multiple_of
                        )
        self.ffn_norm = RMSNormLLaMA(args.dim, eps=args.norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states)) #TODO: check if correct
        return self.ffn_norm(hidden_states + input_tensor) #TODO: check if needed
        #return (hidden_states+input_tensor)
    
class CrossAttentionBlockLLaMA(nn.Module):
    """LLaMA cross attention layer.
    Based on: LLaMA
    """
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention_x = CrossAttentionLLaMA(args)
        self.attention_y = CrossAttentionLLaMA(args)
        self.attention_norm = RMSNormLLaMA(args.dim, eps=args.norm_eps)
        self.output_x = LLaMAOutput(args)
        self.output_y = LLaMAOutput(args)

    def forward(self, x, y, start_pos: int, freqs_cis=None, mask=None):
        out_x = self.attention_x.forward(self.attention_norm(x), self.attention_norm(y), start_pos=start_pos , freqs_cis=freqs_cis, mask=mask) #FIXME
        out_y = self.attention_y.forward(self.attention_norm(y), self.attention_norm(x), start_pos=start_pos, freqs_cis=freqs_cis, mask=mask) #FIXME
        return self.output_x(out_x, x), self.output_y(out_y, y)
    

class CrossAttentionLayerBert(nn.Module):
    """BERT cross attention layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.attention_x = CrossAttentionBert(config)
        self.attention_y = CrossAttentionBert(config)
        self.output_x = BertOutput(config)
        self.output_y = BertOutput(config)

    def forward(self, x, y):
        out_x = self.attention_x(x, y)
        out_y = self.attention_y(y, x)
        return self.output_x(out_x, x), self.output_y(out_y, y)
    
class PoolerLLaMA(nn.Module):
    """LLaMA pooler layer.
    Based on: LLaMA
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(in_features, out_features) #lambda x: x #TODO: add hidden_size ?
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        print("first_token_tensor.shape",first_token_tensor.shape)    #TODO: delete
        print("first_token_tensor.type()",first_token_tensor.type())    #TODO: delete type() ?? 
        pooled_output = self.dense(first_token_tensor)
        print("pooled_output.shape",pooled_output.shape)    #TODO: delete
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PoolerBert(nn.Module):
    """BERT pooler layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        print("Pooler(): first_token_tensor_shape", first_token_tensor.shape)    #TODO: delete
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MultiModalTransformerLLaMA(LLaMAPretrainedModel):
    """
    Multimodal Tranformer with pretrained weights from LLaMA.
    for image text pairs.
    """
    
    def __init__(
        self, num_vision_layers: int, num_text_layers: int, num_cross_attention_layers: int, llama_config: dict, model_args: ModelArgs
    ) -> None:
        """
        Args:
            num_vision_layers: number of vision transformer layers.
            num_text_layers: number of text transformer layers.
            num_cross_attention_layers: number of layer used for mutual information (cross attention).
            llama_config: configuration for LLaMA transformer
        """
        super().__init__(model_args) #here the model_args are passed to the Transformer from llama
        self.config = type("obj", (object,), llama_config)
        assert self.config.num_embeddings != -1 #must have been set correctly by tokenizer in LLaMAPretrainedModel by "super().__init__()"
        self.embeddings = ParallelEmbedding(num_embeddings=self.config.num_embeddings, 
                                            embedding_dim=self.config.embedding_dim, init_method=lambda x: x)
        params = {  "dim": llama_config["embedding_dim"],
                    "multiple_of": llama_config["multiple_of"],
                    "n_heads": llama_config["num_attention_heads"],
                    "n_layers": llama_config["n_layers"],
                    "norm_eps": llama_config["norm_eps"],
                    "vocab_size": llama_config["num_embeddings"],
                    "max_seq_len": llama_config["max_seq_len"],
                    "max_batch_size": llama_config["max_batch_size"],
        }

        model_args: ModelArgs = ModelArgs(**params)

        self.norm = RMSNormLLaMA(model_args.dim, eps=model_args.norm_eps)
        self.output = ColumnParallelLinear(
            model_args.dim, model_args.vocab_size, bias=False, init_method=lambda x: x
        )
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.text_encoder = nn.ModuleList([TransformerBlock_LLaMA(layer, model_args) for layer in range(num_text_layers)])
        self.vision_encoder = nn.ModuleList([TransformerBlock_LLaMA(layer, model_args) for layer in range(num_vision_layers)])
        self.cross_attention_encoder = nn.ModuleList([CrossAttentionBlockLLaMA(model_args) for _ in range(num_cross_attention_layers)]) #FIXME: check if parallel even works or other functions must be implemented for llama
        self.apply(self.init_LLaMA_weights)

    def forward(self, input_ids, start_pos: int, token_type_ids=None, vision_features=None, attention_mask=None):
        _bsz, seqlen = input_ids.shape
        text_features = self.embeddings(input_ids,) #token_type_ids (buffer) not used here
        #print("text_features_type", text_features.type())
        #print("text_features_shape", text_features.shape)
        if vision_features is not None and vision_features.type() is not torch.cuda.HalfTensor:
            vision_features = vision_features.half()
            print("vision_features_type was changed from torch.cuda.FloatTensor to", vision_features.type())
        #print("text_features", text_features)
        self.freqs_cis_text = self.freqs_cis.to(text_features.device)
        freqs_cis_text = self.freqs_cis[start_pos : start_pos + seqlen]
        self.freqs_cis_vision = self.freqs_cis.to(vision_features.device)
        freqs_cis_vision = self.freqs_cis[start_pos : start_pos + vision_features.shape[1]]   #FIXME: check if how 64 can be replaced by seqlen

        """ # see also Transformer forward method for further information
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        """
        
        for layer in self.text_encoder:
            print("-----------")
            print("text_features_shape", text_features.shape)
            print("text_features", text_features)
            text_features = layer(text_features, start_pos, freqs_cis_text, mask=attention_mask) #[0] ... was wrong

        for layer in self.vision_encoder:
            print("-----------")
            print("vision_features_shape", vision_features.shape)
            print("vision_features", vision_features)
            vision_features = layer(vision_features, start_pos, freqs_cis_vision, mask=None) #[0] mask must be None
        for layer in self.cross_attention_encoder:
            print("-----------")
            print("vision_features_shape", vision_features.shape)
            print("text_features_shape", text_features.shape)
            vision_features, text_features = layer(vision_features, text_features, start_pos, freqs_cis=None, mask=None) #FIXME: check freqs_cis

        vision_features = self.norm(vision_features)
        text_features = self.norm(text_features)
        print("vision_features.type()", vision_features.type())
        print("vision_features_shape", vision_features.shape)
        print("text_features_shape", text_features.shape)
        print("vision_features[:, -1, :].shape", vision_features[:, -1, :].shape)
        print("text_features[:, -1, :].shape", text_features[:, -1, :].shape)
        output_vision = self.output(vision_features[:, -1, :])  # only compute last logits
        output_text = self.output(text_features[:, -1, :])  # only compute last logits
        
        print("output_vision.txpe()", output_text.type())
        print("output_vision_shape", output_vision.shape)
        print("output_text_shape", output_text.shape)
        print("output_text", output_text)

        return output_vision, output_text

class MultiModalTransformerBert(BertPreTrainedModel):
    """
    Multimodal Tranformer with pretrained weights from BERT
    for image text pairs.
    """
    
    def __init__(
        self, num_vision_layers: int, num_text_layers: int, num_cross_attention_layers: int, bert_config: dict
    ) -> None:
        """
        Args:
            num_vision_layers: number of vision transformer layers.
            num_text_layers: number of text transformer layers.
            num_cross_attention_layers: number of layer used for mutual information (cross attention).
            bert_config: configuration for bert text transformer
        """
        super().__init__()
        self.config = type("obj", (object,), bert_config)
        self.embeddings = BertEmbeddings(self.config)
        self.text_encoder = nn.ModuleList([BertLayer(self.config) for _ in range(num_text_layers)])
        self.vision_encoder = nn.ModuleList([BertLayer(self.config) for _ in range(num_vision_layers)])
        self.cross_attention_encoder = nn.ModuleList([CrossAttentionLayerBert(self.config) for _ in range(num_cross_attention_layers)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, start_pos=None, token_type_ids=None, vision_features=None, attention_mask=None):
        text_features = self.embeddings(input_ids, token_type_ids)
        for layer in self.vision_encoder:
            vision_features = layer(vision_features, None)[0]
        for layer in self.text_encoder:
            text_features = layer(text_features, attention_mask)[0]
        for layer in self.cross_attention_encoder:
            vision_features, text_features = layer(vision_features, text_features)
        return vision_features, text_features


class ImageTextNet(torch.nn.Module):
    """
    ImageTextNet is a multimodal Transformer for Vision and Language.
    The architecture contains a Vision Transformer (ViT) for image input and a BERT for text.
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
        llama_path: str, #added
        spatial_dims: int = 3,
        hidden_size: int = 768,
        drop_out: float = 0.0,
        attention_probs_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 512,
        model_type: str = "bert", #possible options: LLaMA7B, LLaMA13B, LLaMA30B, bert
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        transformers_version: str = "4.10.2",
        type_vocab_size: int = 2,
        use_cache: bool = True,
        vocab_size: int = 32768,
        dim: int = 4096,  #added, used for embedding_dim in ParallelEmbedding class
        multiple_of = 256, #added, used for multiple_of in TransformerBlock class
        chunk_size_feed_forward: int = 0,
        is_decoder: bool = False,
        add_cross_attention: bool = False
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
            spatial_dims: dimension of spatial size.
            hidden_size: ......
            drop_out: fraction of the input units to drop.
        
        The other parameters are part of the 'bert_config' to 'MultiModal.from_pretrained'.
        """
        
        super().__init__()
        bert_config = {
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "classifier_dropout": None,
            "gradient_checkpointing": gradient_checkpointing,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "hidden_size": hidden_size,
            "initializer_range": initializer_range,
            "intermediate_size": intermediate_size,
            "layer_norm_eps": layer_norm_eps,
            "max_position_embeddings": max_position_embeddings,
            "model_type": model_type,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "pad_token_id": pad_token_id,
            "position_embedding_type": position_embedding_type,
            "transformers_version": transformers_version,
            "type_vocab_size": type_vocab_size,
            "use_cache": use_cache,
            "vocab_size": vocab_size,
            "chunk_size_feed_forward": chunk_size_feed_forward,
            "is_decoder": is_decoder,
            "add_cross_attention": add_cross_attention
        }
        llama_config = {
            "model_path": llama_path,
            "model_type": model_type,
            "num_embeddings": vocab_size, #TODO: set the following vars correctly here
            "embedding_dim": dim,
            "multiple_of": multiple_of,#TODO: add var here
            "num_attention_heads": num_attention_heads, #TODO: check if correct or additional var needed for n_heads
            "n_layers": num_hidden_layers, #TODO: --||--
            "norm_eps": 1e-6,
            "max_seq_len": 512,
            "max_batch_size": 32,
            "hidden_size": dim,     #FIXME: hidden_size must be equal to dim, make more clearly
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "initializer_range": initializer_range,
            #...
            #
        }
        if not (0 <= drop_out <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if (m < p):
                raise ValueError("patch_size should be smaller than img_size.")

            if (m % p != 0):  # type: ignore
                raise ValueError("img_size should be divisible by patch_size.")

        if model_type == "bert":
            self.multimodal = MultiModalTransformerBert.from_pretrained(
            num_vision_layers=num_vision_layers,
            num_text_layers=num_text_layers,
            num_cross_attention_layers=num_cross_attention_layers,
            bert_config=bert_config
            )
            self.pooler = PoolerBert(hidden_size=hidden_size)
            self.cls_head = torch.nn.Linear(hidden_size, num_classes)
        elif model_type == "LLaMA7B" or model_type == "LLaMA13B" or model_type == "LLaMA30B":
            if hidden_size != dim: 
                hidden_size=dim
                print(f"Warning: hidden_size was set to {hidden_size} to match embedding_dim.")
            self.multimodal = MultiModalTransformerLLaMA.from_pretrained(
            num_vision_layers=num_vision_layers,
            num_text_layers=num_text_layers,
            num_cross_attention_layers=num_cross_attention_layers,
            llama_config=llama_config
            )
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.pooler = PoolerLLaMA(llama_config["num_embeddings"], dim) #TODO: check if correct
            self.cls_head = ColumnParallelLinear(hidden_size, num_classes) #torch.nn.Linear(hidden_size, num_classes)#FIXME
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            raise ValueError("model_type should be either 'bert', 'LLaMA7B' or 'LLaMA13B'.")
        
        self.patch_size = patch_size
        self.num_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, self.patch_size)])
        self.vision_proj: nn.Module
        #Convolution
        #factory layer: applies one of "Conv1D, Conv2D, Conv3D" functions depending on the spatial_dims
        #               see also "ViT(MONAI)" for further information
        self.vision_proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=hidden_size, kernel_size=self.patch_size, stride=self.patch_size
        )
        #TODO: perceptron ... not implemented (yet) -- not needed?
        
        self.norm_vision_pos = nn.LayerNorm(hidden_size)
        self.pos_embed_vis = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.drop = torch.nn.Dropout(drop_out)
        
    def forward(self, input_ids, token_type_ids=None, vision_features=None):   ##__call__()
        #token_type_ids ... buffer
        #TODO: start_pos must be added here... #used for llama
        start_pos = 0
        print("input_ids", input_ids)
        print("input_ids_type", input_ids.type())
        print("input_ids_shape", input_ids.shape)
        attention_mask = torch.ones_like(input_ids).unsqueeze(1).unsqueeze(2) #reshape tensor to dimension of size 1 at specific position
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0
        #print(f"   vision_features = {vision_features}")
        vision_features = self.vision_proj(vision_features).flatten(2).transpose(1,2) #FIXME: flatten: reshaping it into 1-dimensional tensor
        vision_features = self.norm_vision_pos(vision_features)         #attention: transpose(-1,-2) ???
        #print("vision_features_shape",vision_features.shape)
        #print("self.pos_embed_vis_shape",self.pos_embed_vis.shape)
        vision_features = vision_features + self.pos_embed_vis
        print("vision_features_type", vision_features.type())       
        hidden_state_vision, hidden_state_text = self.multimodal(
            input_ids=input_ids, start_pos=start_pos, token_type_ids=token_type_ids, vision_features=vision_features, attention_mask=attention_mask
        )
        #analyze multimodal output
        print("Pooling...")
        print(f"hidden_state_text.size() = {hidden_state_text.size()}")
        print(f"hidden_state_vision.size() = {hidden_state_vision.size()}")
            #add usage of hidden_state vision to pooler what also affects logits
        pooled_text_features = self.pooler(hidden_state_text) #dense + activation
        pooled_vision_features = self.pooler(hidden_state_vision)
        print(f"pooled_text_features = {pooled_text_features}")
        print(f"pooled_vision_features = {pooled_vision_features}")
        logits = self.drop(pooled_text_features + pooled_vision_features) #TODO: check if correct
        print(f"logits.size() = {logits.size()}")
        logits = self.cls_head(logits) #num_classes outputs in [0,1] (due to tanh() activation in pooling)
        print("logits.size()", logits.size())
        print("logits.type()", logits.type())
        return logits