"""
based on base_cam from pytorch_grad_cam

"""

import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from interpretability.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from torch.cuda.amp import GradScaler 

def scale_cam_list(cam, target_size=None): #copied from pytorch_grad_cam.utils.image import scale_cam_image
    print("cam.shape",cam.shape)    #TODO: delete
    #print("cam", cam)
    print("target_size",target_size)    #TODO: delete
    result = []
    for text in cam:
        text = np.float32(text) #needed in order to use cv2.INTER_LINEAR_EXACT
        text = text - np.min(text)
        text = text / (1e-7 + np.max(text))
        print("text", text)
        if target_size is not None: #TODO: change to float32 ?, or use cv2.INTER_NEAREST
            text = cv2.resize(text, target_size, interpolation=cv2.INTER_LINEAR_EXACT) #interpolation=cv2.INTER_LINEAR_EXACT, INTER_NEAREST
        result.append(text)
    result = np.float32(result)
    return result

class BaseCAMText:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.scaler = GradScaler() #TODO: ERROR: not callable

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor_text: torch.Tensor,
                        input_tensor_vision: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_list(self,
                      input_tensor_text: torch.Tensor,
                      input_tensor_vision: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor_text,
                                       input_tensor_vision,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        print("weights.shape", weights.shape)    #TODO: delete
        print(weights)
        weighted_activations = weights[:, :, None] * activations
        print("weighted_activations.shape" ,weighted_activations.shape)    #TODO: delete
        cam = weighted_activations.sum(axis=2)
        print("cam.shape", cam.shape)    #TODO: delete
        return cam

    def forward(self,
                input_tensor_text: torch.Tensor,
                input_tensor_vision: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False  #TODO: needed?
                ) -> np.ndarray:

        print("\n\n\nForwarding text CAM works properly\n\n\n")

        if self.cuda:
            input_tensor_text = input_tensor_text.cuda()
            input_tensor_vision = input_tensor_vision.cuda()

        if self.compute_input_gradient:
            input_tensor_text = torch.autograd.Variable(input_tensor_text,
                                                        requires_grad=True)
            input_tensor_vision = torch.autograd.Variable(input_tensor_vision,
                                                          requires_grad=True)

        outputs = self.activations_and_grads(input_tensor_text, #model(x,y) forward() call
                                             input_tensor_vision)

        #replaced:(if targets is None:) #we always compute the targets from the outputs !!!
        print("outputs", outputs)    #TODO: delete
        target_categories = [1 if (p > 0.5) else 0 for p in outputs[0]]
        #get position where 1 occurs
        target_categories = [i for i, x in enumerate(target_categories) if x == 1]
        #handle error when no category is predicted: take argmax
        if sum(target_categories) == 0:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        print("\n\ntarget_categories\n\n", target_categories)    #TODO: delete
        targets = [ClassifierOutputTarget(
            category) for category in target_categories]
        print("\n\ntargets\n\n", targets)    #TODO: delete
        print("Targets overwritten!")    #TODO: delete

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            self.scaler.scale(loss).backward(retain_graph=True)
            #update scaler
            #self.scaler.update() #TODO: check if and where needed

        cam_per_layer, cam_per_layer_unscaled = self.compute_cam_per_layer(
                                                input_tensor_text,
                                                input_tensor_vision,
                                                targets,
                                                eigen_smooth,
                                                )
        return self.aggregate_multi_layers(cam_per_layer), self.aggregate_multi_layers(cam_per_layer_unscaled, scale=False)

    """ def get_target_width_height(self,
                                input_tensor_vision: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor_vision.size(-1), input_tensor_vision.size(-2)
        return width, height
    """ # not needed for text

    def compute_cam_per_layer(
            self,
            input_tensor_text: torch.Tensor,
            input_tensor_vision: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = None  #as not needed here when analyzing text

        cam_per_target_layer = []
        cam_per_target_layer_unscaled = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_list(input_tensor_text,
                                     input_tensor_vision,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            print("cam", cam)    #TODO: delete
            cam = np.maximum(cam, 0)
            scaled = scale_cam_list(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer_unscaled.append(cam[:, None, :])

        return cam_per_target_layer, cam_per_target_layer_unscaled

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray,
            scale = True) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        if scale:
            return scale_cam_list(result)
        else:
            return result

    """
    def forward_augmentation_smoothing(self,
                                       input_tensor_vision: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor_vision)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam
    """

    def __call__(self,
                 input_tensor_text: torch.Tensor,
                 input_tensor_vision: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        """
        if aug_smooth is True:  #does not make much sense here.. but irrelevant if applied when analyzing text
            return self.forward_augmentation_smoothing(
                input_tensor_vision, targets, eigen_smooth)
        """
        return self.forward(input_tensor_text,
                            input_tensor_vision,
                            targets, eigen_smooth)


    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

#end of base_cam.py