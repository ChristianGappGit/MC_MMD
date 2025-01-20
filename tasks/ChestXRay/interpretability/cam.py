"""
based on cam from pytorch_grad_cam
"""

from typing import List
import numpy as np
import torch
from torch.nn.modules import Module
from interpretability_ChestXRay.base_cam import BaseCAM
from interpretability_ChestXRay.base_cam_text import BaseCAMText
from interpretability_ChestXRay.colorize_text import colorize_tokens2

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor_text,
                        input_tensor_vision,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        print("input_tensor_text.shape",input_tensor_text.shape)    #TODO: delete
        print("input_tensor_vision.shape",input_tensor_vision.shape)    #TODO: delete
        print("target_layer",target_layer)    #TODO: delete
        print("target_category",target_category)    #TODO: delete
        print("activations.shape",activations.shape)    #TODO: delete
        print("grads.shape",grads.shape)    #TODO: delete
        return np.mean(grads, axis=(2, 3))  #TODO: check if this is correct

class GradCAMText(BaseCAMText):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None, color_map='turbo', map_colors_to_min_max=True
                ):
        self.color_map = color_map
        self.map_colors_to_min_max = map_colors_to_min_max
        self.mapping_range = [0.15,0.85]
        super(GradCAMText,
                self).__init__(
                    model, 
                    target_layers, 
                    use_cuda, 
                    reshape_transform)

    def __call__(self, input_tensor_text, input_tensor_vision, targets, input_tokens
                ):
        """
        input_tensor_text: tokenized text
            [[...]], example: [[101, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, ...]]
        input_tensor_vision: image (feature extraction must be done by "model")

        targets: target classes
            [...]
        input_tokens: text tokens
            [[], [], ... ] example: [['F'], ['##IND'], ['##ING'], ['##S'], [':'], ['PA'], ['and'], ['later'], ['##al'], ['views'],...]

        returns: colorized text and color bar as html string
        """
        token_map, token_map_unscaled = super(GradCAMText, self).__call__(input_tensor_text, input_tensor_vision, targets)  #TODO: check if works properly
        #first dereference token_map once
        print("token_map.shape", token_map.shape)    #TODO: delete
        print("input_tokens", input_tokens)    #TODO: delete
        if isinstance(token_map, torch.Tensor):
            token_map = token_map.detach().cpu().numpy()
        if isinstance(token_map_unscaled, torch.Tensor):
            token_map_unscaled = token_map_unscaled.detach().cpu().numpy()
        token_map = token_map[0] if token_map.shape[0] == 1 else token_map
        token_map_unscaled = token_map_unscaled[0] if token_map_unscaled.shape[0] == 1 else token_map_unscaled
        #map to min max range:
        if self.map_colors_to_min_max:
            token_map = (token_map - np.min(token_map)) / (np.max(token_map) - np.min(token_map)+1e-7)
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.detach().cpu().numpy()
        input_tokens = input_tokens[0] if len(input_tokens) == 1 else input_tokens
        input_tokens = [item for sublist in input_tokens for item in sublist]
        print("token_map.shape", token_map.shape)    #TODO: delete
        print("token_map (=color_array)", token_map)    #TODO: delete
        print("len(input_tokens)", len(input_tokens))    #TODO: delete
        print("input_tokens", input_tokens)    #TODO: delete
        
        assert len(input_tokens) == len(token_map)

        colorized_text, color_bar = colorize_tokens2(tokens=input_tokens,
                                        color_array=token_map,
                                        token_concat='##',
                                        skip_tokens=["**NULL**"],
                                        color_map=self.color_map,
                                        mapping_range=self.mapping_range)
        return colorized_text, color_bar, np.sum(abs(token_map_unscaled))

    def get_cam_weights(self,
                        input_tensor_text,
                        input_tensor_vision,
                        target_layer,
                        target_category,
                        activations,
                        grads):
            print("-------------------GradCAMText-------------------")
            print("input_tensor_text.shape",input_tensor_text.shape)    #TODO: delete
            print("input_tensor_vision.shape",input_tensor_vision.shape)    #TODO: delete
            print("target_layer",target_layer)    #TODO: delete
            print("target_category",target_category)    #TODO: delete
            print("activations.shape",activations.shape)    #TODO: delete
            print("grads.shape",grads.shape)    #TODO: delete
            print("grads",grads)    #TODO: delete
            return np.mean(grads, axis=(2))  #TODO: check if this is correct
    
class GradCAMCross():
    """
    To be implemented
    """

    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None,
            color_map='turbo', map_colors_to_min_max=True):
        pass

    def __call__(self, input_tensor_text, input_tensor_vision, targets, input_tokens):
        pass