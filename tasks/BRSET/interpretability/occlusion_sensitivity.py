#adapted from monai

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.data.meta_tensor import MetaTensor
from monai.networks.utils import eval_mode
from monai.transforms import Compose, GaussianSmooth, Lambda, ScaleIntensity, SpatialCrop
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.visualize.visualizer import default_upsampler
import tqdm

from interpretability.colorize_text import colorize_tokens, colorize_tokens2
from interpretability.colorize_img import colorize_img


class OcclusionSensitivity:
    """
    This class computes the occlusion sensitivity for a model's prediction of a given image. By occlusion sensitivity,
    we mean how the probability of a given prediction changes as the occluded section of an image changes. This can be
    useful to understand why a network is making certain decisions.

    As important parts of the image are occluded, the probability of classifying the image correctly will decrease.
    Hence, more negative values imply the corresponding occluded volume was more important in the decision process.

    Two ``torch.Tensor`` will be returned by the ``__call__`` method: an occlusion map and an image of the most probable
    class. Both images will be cropped if a bounding box used, but voxel sizes will always match the input.

    The occlusion map shows the inference probabilities when the corresponding part of the image is occluded. Hence,
    more -ve values imply that region was important in the decision process. The map will have shape ``BCHW(D)N``,
    where ``N`` is the number of classes to be inferred by the network. Hence, the occlusion for class ``i`` can
    be seen with ``map[...,i]``.

    The most probable class is an image of the probable class when the corresponding part of the image is occluded
    (equivalent to ``occ_map.argmax(dim=-1)``).

    See: R. R. Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization. https://doi.org/10.1109/ICCV.2017.74.

    Examples:

    .. code-block:: python

        # densenet 2d
        from monai.networks.nets import DenseNet121
        from monai.visualize import OcclusionSensitivity
        import torch

        model_2d = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        occ_sens = OcclusionSensitivity(nn_module=model_2d)
        occ_map, most_probable_class = occ_sens(x=torch.rand((1, 1, 48, 64)), b_box=[2, 40, 1, 62])

        # densenet 3d
        from monai.networks.nets import DenseNet
        from monai.visualize import OcclusionSensitivity

        model_3d = DenseNet(spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,))
        occ_sens = OcclusionSensitivity(nn_module=model_3d, n_batch=10)
        occ_map, most_probable_class = occ_sens(torch.rand(1, 1, 6, 6, 6), b_box=[1, 3, -1, -1, -1, -1])

    See Also:

        - :py:class:`monai.visualize.occlusion_sensitivity.OcclusionSensitivity.`
    """

    @deprecated_arg(
        name="pad_val",
        since="1.0",
        removed="1.2",
        msg_suffix="Please use `mode`. For backwards compatibility, use `mode=mean_img`.",
    )
    @deprecated_arg(name="stride", since="1.0", removed="1.2", msg_suffix="Please use `overlap`.")
    @deprecated_arg(name="per_channel", since="1.0", removed="1.2")
    @deprecated_arg(name="upsampler", since="1.0", removed="1.2")
    def __init__(
        self,
        nn_module: nn.Module,
        pad_val: Optional[float] = None,
        mask_size: Union[int, Sequence] = 16,
        n_batch: int = 16,
        stride: Union[int, Sequence] = 1,
        per_channel: bool = True,
        upsampler: Optional[Callable] = default_upsampler,
        verbose: bool = True,
        mode: Union[str, float, Callable] = "gaussian",
        overlap: float = 0.25,
        activate: Union[bool, Callable] = True,
    ) -> None:
        """
        Occlusion sensitivity constructor.

        Args:
            nn_module: Classification model to use for inference
            mask_size: Size of box to be occluded, centred on the central voxel. If a single number
                is given, this is used for all dimensions. If a sequence is given, this is used for each dimension
                individually.
            n_batch: Number of images in a batch for inference.
            verbose: Use progress bar (if ``tqdm`` available).
            mode: what should the occluded region be replaced with? If a float is given, that value will be used
                throughout the occlusion. Else, ``gaussian``, ``mean_img`` and ``mean_patch`` can be supplied:

                * ``gaussian``: occluded region is multiplied by 1 - gaussian kernel. In this fashion, the occlusion
                  will be 0 at the center and will be unchanged towards the edges, varying smoothly between. When
                  gaussian is used, a weighted average will be used to combine overlapping regions. This will be
                  done using the gaussian (not 1-gaussian) as occluded regions count more.
                * ``mean_patch``: occluded region will be replaced with the mean of occluded region.
                * ``mean_img``: occluded region will be replaced with the mean of the whole image.

            overlap: overlap between inferred regions. Should be in range 0<=x<1.
            activate: if ``True``, do softmax activation if num_channels > 1 else do ``sigmoid``. If ``False``, don't do any
                activation. If ``callable``, use callable on inferred outputs.

        """
        self.nn_module = nn_module
        self.mask_size = mask_size
        self.n_batch = n_batch
        self.verbose = verbose
        self.overlap = overlap
        self.activate = activate
        # mode
        if isinstance(mode, str) and mode not in ("gaussian", "mean_patch", "mean_img"):
            raise NotImplementedError
        self.mode = mode

    @staticmethod
    def constant_occlusion(x: torch.Tensor, val: float, mask_size: Sequence) -> Tuple[float, torch.Tensor]:
        """Occlude with a constant occlusion. Multiplicative is zero, additive is constant value."""
        ones = torch.ones((*x.shape[:2], *mask_size), device=x.device, dtype=x.dtype)
        return 0, ones * val

    @staticmethod
    def gaussian_occlusion(x: torch.Tensor, mask_size, sigma=0.25) -> Tuple[torch.Tensor, float]:
        """
        For Gaussian occlusion, Multiplicative is 1-Gaussian, additive is zero.
        Default sigma of 0.25 empirically shown to give reasonable kernel, see here:
        https://github.com/Project-MONAI/MONAI/pull/5230#discussion_r984520714.
        """
        kernel = torch.zeros((x.shape[1], *mask_size), device=x.device, dtype=x.dtype)
        spatial_shape = kernel.shape[1:]
        # all channels (as occluded shape already takes into account per_channel), center in spatial dimensions
        center = [slice(None)] + [slice(s // 2, s // 2 + 1) for s in spatial_shape]
        # place value of 1 at center
        kernel[center] = 1.0
        # Smooth with sigma equal to quarter of image, flip +ve/-ve so largest values are at edge
        # and smallest at center. Scale to [0, 1].
        gaussian = Compose(
            [GaussianSmooth(sigma=[b * sigma for b in spatial_shape]), Lambda(lambda x: -x), ScaleIntensity()]
        )
        # transform and add batch
        mul: torch.Tensor = gaussian(kernel)[None]

        return mul, 0

    @staticmethod
    def predictor(
        cropped_grid: torch.Tensor,
        nn_module: nn.Module,
        x: torch.Tensor, #image
        x_text: torch.Tensor, #text
        mul: Union[torch.Tensor, float],
        add: Union[torch.Tensor, float],
        mask_size: Sequence,
        occ_mode: str,
        activate: Union[bool, Callable],
        module_kwargs,
    ) -> torch.Tensor:
        """
        Predictor function to be passed to the sliding window inferer. Takes a cropped meshgrid,
        referring to the coordinates in the input image. We use the index of the top-left corner
        in combination ``mask_size`` to figure out which region of the image is to be occluded. The
        occlusion is performed on the original image, ``x``, using ``cropped_region * mul + add``. ``mul``
        and ``add`` are sometimes pre-computed (e.g., a constant Gaussian blur), or they are
        sometimes calculated on the fly (e.g., the mean of the occluded patch). For this reason
        ``occ_mode`` is given. Lastly, ``activate`` is used to activate after each call of the model.

        Args:
            cropped_grid: subsection of the meshgrid, where each voxel refers to the coordinate of
                the input image. The meshgrid is created by the ``OcclusionSensitivity`` class, and
                the generation of the subset is determined by ``sliding_window_inference``.
            nn_module: module to call on data.
            x: the image that was originally passed into ``OcclusionSensitivity.__call__``.
            x_text: the text that was originally passed into ``OcclusionSensitivity.__call__``.
            mul: occluded region will be multiplied by this. Can be ``torch.Tensor`` or ``float``.
            add: after multiplication, this is added to the occluded region. Can be ``torch.Tensor`` or ``float``.
            mask_size: Size of box to be occluded, centred on the central voxel. Should be
                a sequence, one value for each spatial dimension.
            occ_mode: might be used to calculate ``mul`` and ``add`` on the fly.
            activate: if ``True``, do softmax activation if num_channels > 1 else do ``sigmoid``. If ``False``, don't do any
                activation. If ``callable``, use callable on inferred outputs.
            module_kwargs: kwargs to be passed onto module when inferring
        """
        n_batch = cropped_grid.shape[0]
        sd = cropped_grid.ndim - 2
        # start with copies of x to infer
        im = torch.repeat_interleave(x, n_batch, 0)
        # get coordinates of top left corner of occluded region (possible because we use meshgrid)
        corner_coord_slices = [slice(None)] * 2 + [slice(1)] * sd
        top_corners = cropped_grid[corner_coord_slices]

        # replace occluded regions
        for b, t in enumerate(top_corners):
            # starting from corner, get the slices to extract the occluded region from the image
            slices = [slice(b, b + 1), slice(None)] + [slice(int(j), int(j) + m) for j, m in zip(t, mask_size)]
            to_occlude = im[slices]
            if occ_mode == "mean_patch":
                add, mul = OcclusionSensitivity.constant_occlusion(x, to_occlude.mean().item(), mask_size)

            if callable(occ_mode):
                to_occlude = occ_mode(x, to_occlude)
            else:
                to_occlude = to_occlude * mul + add
            if add is None or mul is None:
                raise RuntimeError("Shouldn't be here, something's gone wrong...")
            im[slices] = to_occlude
        # infer
        out: torch.Tensor = nn_module(x_text, im, **module_kwargs)

        # if activation is callable, call it
        if callable(activate):
            out = activate(out)
        # else if True (should be boolean), sigmoid if n_chan == 1 else softmax
        elif activate:
            out = out.sigmoid() if x.shape[1] == 1 else out.softmax(1)

        # the output will have shape [B,C] where C is number of channels output by model (inference classes)
        # we need to return it to sliding window inference with shape [B,C,H,W,[D]], so add dims and repeat values
        for m in mask_size:
            out = torch.repeat_interleave(out.unsqueeze(-1), m, dim=-1)

        return out

    @staticmethod
    def crop_meshgrid(
        grid: MetaTensor, b_box: Sequence, mask_size: Sequence
    ) -> Tuple[MetaTensor, SpatialCrop, Sequence]:
        """Crop the meshgrid so we only perform occlusion sensitivity on a subsection of the image."""
        # distance from center of mask to edge is -1 // 2.
        mask_edge = [(m - 1) // 2 for m in mask_size]
        bbox_min = [max(b - m, 0) for b, m in zip(b_box[::2], mask_edge)]
        bbox_max = []
        for b, m, s in zip(b_box[1::2], mask_edge, grid.shape[2:]):
            # if bbox is -ve for that dimension, no cropping so use current image size
            if b == -1:
                bbox_max.append(s)
            # else bounding box plus distance to mask edge. Make sure it's not bigger than the size of the image
            else:
                bbox_max.append(min(b + m, s))
        # bbox_max = [min(b + m, s) if b >= 0 else s for b, m, s in zip(b_box[1::2], mask_edge, grid.shape[2:])]
        # No need for batch and channel slices. Batch will be removed and added back in, and
        # SpatialCrop doesn't act on the first dimension anyway.
        slices = [slice(s, e) for s, e in zip(bbox_min, bbox_max)]
        cropper = SpatialCrop(roi_slices=slices)
        cropped: MetaTensor = cropper(grid[0])[None]  # type: ignore
        mask_size = list(mask_size)
        for i, s in enumerate(cropped.shape[2:]):
            mask_size[i] = min(s, mask_size[i])
        return cropped, cropper, mask_size

    def __call__(
        self, x: torch.Tensor, x_text: torch.Tensor, b_box: Optional[Sequence] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image to use for inference. Should be a tensor consisting of 1 batch.
            x_text: Text to use for inference. Should be a tensor consisting of 1 batch.
            b_box: Bounding box on which to perform the analysis. The output image will be limited to this size.
                There should be a minimum and maximum for all spatial dimensions: ``[min1, max1, min2, max2,...]``.
                * By default, the whole image will be used. Decreasing the size will speed the analysis up, which might
                    be useful for larger images.
                * Min and max are inclusive, so ``[0, 63, ...]`` will have size ``(64, ...)``.
                * Use -ve to use ``min=0`` and ``max=im.shape[x]-1`` for xth dimension.
                * N.B.: we add half of the mask size to the bounding box to ensure that the region of interest has a
                    sufficiently large area surrounding it.
            kwargs: any extra arguments to be passed on to the module as part of its `__call__`.

        Returns:
            * Occlusion map:
                * Shows the inference probabilities when the corresponding part of the image is occluded.
                    Hence, more -ve values imply that region was important in the decision process.
                * The map will have shape ``BCHW(D)N``, where N is the number of classes to be inferred by the
                    network. Hence, the occlusion for class ``i`` can be seen with ``map[...,i]``.
                * If `per_channel==False`, output ``C`` will equal 1: ``B1HW(D)N``
            * Most probable class:
                * The most probable class when the corresponding part of the image is occluded (``argmax(dim=-1)``).
            Both images will be cropped if a bounding box used, but voxel sizes will always match the input.
        """
        if x.shape[0] > 1:
            raise ValueError("Expected batch size of 1.")

        sd = x.ndim - 2
        mask_size: Sequence = ensure_tuple_rep(self.mask_size, sd)

        # get the meshgrid (so that sliding_window_inference can tell us which bit to occlude)
        grid: MetaTensor = MetaTensor(
            np.stack(np.meshgrid(*[np.arange(0, i) for i in x.shape[2:]], indexing="ij"))[None],
            device=x.device,
            dtype=x.dtype,
        )
        # if bounding box given, crop the grid to only infer subsections of the image
        if b_box is not None:
            grid, cropper, mask_size = self.crop_meshgrid(grid, b_box, mask_size)

        # check that the grid is bigger than the mask size
        if any(m > g for g, m in zip(grid.shape[2:], mask_size)):
            raise ValueError(f"Image (spatial shape) {grid.shape[2:]} should be bigger than mask {mask_size}.")

        # get additive and multiplicative factors if they are unchanged for all patches (i.e., not mean_patch)
        add: Optional[Union[float, torch.Tensor]]
        mul: Optional[Union[float, torch.Tensor]]
        # multiply by 0, add value
        if isinstance(self.mode, float):
            mul, add = self.constant_occlusion(x, self.mode, mask_size)
        # multiply by 0, add mean of image
        elif self.mode == "mean_img":
            mul, add = self.constant_occlusion(x, x.mean().item(), mask_size)
        # for gaussian, additive = 0, multiplicative = gaussian
        elif self.mode == "gaussian":
            mul, add = self.gaussian_occlusion(x, mask_size)
        # else will be determined on each patch individually so calculated later
        else:
            add, mul = None, None

        with eval_mode(self.nn_module):
            # needs to go here to avoid circular import
            #from monai.inferers import sliding_window_inference    
            from interpretability.utils import sliding_window_inference #adapted version of monai's sliding_window_inference

            sensitivity_im: MetaTensor = sliding_window_inference(  # type: ignore
                grid,
                roi_size=mask_size,
                sw_batch_size=self.n_batch,
                predictor=OcclusionSensitivity.predictor,
                overlap=self.overlap,
                mode="gaussian" if self.mode == "gaussian" else "constant",
                progress=self.verbose,
                nn_module=self.nn_module,
                x=x,
                x_text=x_text,
                add=add,
                mul=mul,
                mask_size=mask_size,
                occ_mode=self.mode,
                activate=self.activate,
                module_kwargs=kwargs,
            )

        if b_box is not None:
            # undo the cropping that was applied to the meshgrid
            sensitivity_im = cropper.inverse(sensitivity_im[0])[None]  # type: ignore
            # crop using the bounding box (ignoring the mask size this time)
            bbox_min = [max(b, 0) for b in b_box[::2]]
            bbox_max = [b if b > 0 else s for b, s in zip(b_box[1::2], x.shape[2:])]
            cropper = SpatialCrop(roi_start=bbox_min, roi_end=bbox_max)
            sensitivity_im = cropper(sensitivity_im[0])[None]  # type: ignore

        # The most probable class is the max in the classification dimension (1)
        most_probable_class = sensitivity_im.argmax(dim=1, keepdim=True)
        return sensitivity_im, most_probable_class
    

class OcclusionSensitivityText:
    def __init__(self, nn_module, replace_token, skip_token="**NULL**", color_map='turbo', map_colors_to_min_max=True,
                 mask_words = True):
        self.model = nn_module  
        self.color_map = color_map
        self.map_colors_to_min_max = map_colors_to_min_max
        self.mapping_range = [0.15,0.85] # [0.15,0.85] herein the colormap is not too dark for unimportant tokens and thus still readable
        self.replace_token = replace_token
        self.skip_token = skip_token
        self.maks_words = mask_words #True: mask words instead of syllables
        print("self.replace_token", self.replace_token)    #TODO: delete

    def __call__(self, x_text, x_tokens, x_img,
                 ):
        """
        x_text must be a vector of syllables
            [[], []..]
        x_tokens are tokenized syllables (numbers in vectors)
            [[..]]
        x_img is the image
        
        occludes syllables in x_tokens, recomputes the prediction, and returns colorirized text based on the difference in prediction
        syll occlusion is done by replacing the syllable with a special token (i.e. "[MASK]")
        """
        #compute outputs unmasked
        outputs = self.model(x_tokens, x_img)
        outputs = outputs.detach().cpu().numpy()
        outputs_pred = [1 if x >= 0.5 else 0 for x in outputs[0]]

        #for each syllable, replace with self.replace_token, compute outputs masked and store difference

        print("x_tokens", x_tokens)    #TODO: delete
        print("x_text", x_text)
        tqdm, _ = optional_import("tqdm", name="tqdm")

        if self.replace_token is None: #take the mean of the syllables
            #cast x_tokens[0] to float
            x_token_float = x_tokens[0].float()
            print("x_token_float", x_token_float)    #TODO: delete
            #take mean (without zeros) and round
            replace_token = int(torch.mean(x_token_float[x_token_float != 0]).round())
            print("x_token_float", x_token_float)    #TODO: delete
            print("torch.mean(x_token_float)", torch.mean(x_token_float))    #TODO: delete
        else: 
            replace_token = self.replace_token
            
        print("replace_token", replace_token)    #TODO: delete

        n = 0 #number of syllables or words (depending on self.mask_words) that forced a class change

        if self.maks_words: #compute output diffs of words, instead of syllables
            len_word = 0
            occ_sens_class = np.empty((0, outputs.shape[1])) #empty array to store differences
            token_map = np.empty((0, outputs.shape[1])) #empty array to store differences
            for i in tqdm(range(len(x_tokens[0]))):
                if x_text[i][0] == self.skip_token and len_word == 0:
                    continue
                if x_text[i][0].startswith("##"):
                    len_word += 1
                    continue
                if len_word == 0:   #new word, first syllable
                    len_word = 1
                    continue
                #now we have the start of a new word
                #old word is from i-len_word to i, processed in the following lines
                #store replaced word as last len_word syllables
                replaced_word = x_tokens[0][i-len_word:i].clone()
                #replace the tokens now with replace_token
                x_tokens[0][i-len_word:i] = replace_token*torch.ones(len_word, dtype=torch.long)
                #compute outputs masked
                outputs_masked = self.model(x_tokens, x_img)
                outputs_masked = outputs_masked.detach().cpu().numpy()
                #see, if classes changed. If yes, store 1, else 0
                outputs_masked_pred = [1 if x >= 0.5 else 0 for x in outputs_masked[0]]
                occ_sens_class_i = [1 if x != y else 0 for x, y in zip(outputs_pred, outputs_masked_pred)]
                """
                #test: make some noise to data  #TESTED: works!!!
                if i % 3 == 0:
                    occ_sens_class_i[0] = 1 #(change one class)
                """
                if np.sum(occ_sens_class_i) > 0: #if at least one class changed
                    n += 1
                #repeat occ_sens_class_i to len_word
                occ_sens_class_i = np.array([occ_sens_class_i])
                occ_sens_class_i = np.repeat(occ_sens_class_i, len_word, axis=0) #TODO: uncomment
                #concat to occ_sens_class
                occ_sens_class = np.concatenate((occ_sens_class, occ_sens_class_i), axis=0)
                #compute difference
                diff = outputs - outputs_masked #[1 x num_classes]
                #print("diff", diff)    #TODO: delete
                #store diff for each syllable (len_word times) in diff
                diff = np.repeat(diff, len_word, axis=0)
                #print("diff", diff)    #TODO: delete
                #store difference in token_map
                token_map = np.concatenate((token_map, diff), axis=0)
                #restore word in tokens
                x_tokens[0][i-len_word:i] = replaced_word
                if x_text[i][0] == self.skip_token:
                    len_word = 0
                else: 
                    len_word = 1 #reset len_word to 1, as we already have a new word
                #free memory
                del replaced_word, outputs_masked, diff
            del outputs
                
        else: #compute output diffs with syllables
            for i in tqdm(range(len(x_tokens[0]))):
                if x_text[i][0] == self.skip_token:
                    continue
                #replace syllable with replace_token
                replaced_syll = x_tokens[0][i].clone()
                #print("replaced_syll", replaced_syll)    #TODO: delete
                x_tokens[0][i] = replace_token
                #print("x_tokens[0][i]", x_tokens[0][i])    #TODO: delete
                #print("replaced_syll", replaced_syll)    #TODO: delete
                #compute outputs masked
                outputs_masked = self.model(x_tokens, x_img)
                outputs_masked = outputs_masked.detach().cpu().numpy()
                #see, if classes changed. If yes, store 1, else 0
                outputs_masked_pred = [1 if x >= 0.5 else 0 for x in outputs_masked[0]]
                occ_sens_class_i = [1 if x != y else 0 for x, y in zip(outputs_pred, outputs_masked_pred)]
                if np.sum(occ_sens_class_i) > 0: #if at least one class changed
                    n += 1
                occ_sens_class_i = np.array([occ_sens_class_i])
                occ_sens_class = occ_sens_class_i if i == 0 else np.concatenate((occ_sens_class, occ_sens_class_i), axis=0)
                #compute difference
                diff = outputs - outputs_masked #[1 x num_classes]
                #store difference in token_map
                token_map = diff if i == 0 else np.concatenate((token_map, diff), axis=0)
                #restore syllable
                x_tokens[0][i] = replaced_syll
                #free memory
                del replaced_syll, outputs_masked, diff #token_map stays untouched from "del"
            del outputs
        
        #as negative values are more important, we take invert the values
        token_map = -token_map
        #copy of token_map for visualization
        token_map_max = token_map.copy()
        token_map_mean = token_map.copy()
        #token_map is now of shape [len(x_tokens[0]-n(skip_tokens)), num_classes]
        #compute the mean of the difference for each syllable
        token_map_mean = np.mean(token_map_mean, axis=1)
        #compute the max of the difference for each syllable
        token_map_max = np.max(token_map_max, axis=1)
        #see if classes changed. If yes, store 1, else 0
        occ_sens_class = np.sum(occ_sens_class, axis=1) #for each syllable or word
        occ_sens_class[occ_sens_class > 0] = 1

        #map to min max range:
        if self.map_colors_to_min_max:
            token_map_mean = (token_map_mean - np.min(token_map_mean)) / (np.max(token_map_mean) - np.min(token_map_mean)+1e-7)
            token_map_max = (token_map_max - np.min(token_map_max)) / (np.max(token_map_max) - np.min(token_map_max)+1e-7)
        if isinstance(x_text, torch.Tensor):
            x_text = x_text.detach().cpu().numpy()
        x_text = x_text[0] if len(x_text) == 1 else x_text
        x_text = [item for sublist in x_text for item in sublist]
        #print("token_map.shape", token_map.shape)    #TODO: delete
        #print("token_map (=color_array)", token_map)    #TODO: delete
        #print("len(x_text)", len(x_text))    #TODO: delete
        #print("x_text", x_text)    #TODO: delete
        
        #print("occ_sens_class", occ_sens_class)
        #print("token_map", token_map)
        assert len(token_map) == len(token_map_max) == len(token_map_mean) == len(occ_sens_class)
        colorized_text_mean, color_bar_mean = colorize_tokens2(tokens=x_text, 
                                color_array=token_map_mean, 
                                token_concat='##',
                                skip_tokens=["**NULL**"],
                                color_map=self.color_map,
                                mapping_range=self.mapping_range,
                                )
        colorized_text_max, color_bar_max = colorize_tokens2(tokens=x_text, 
                                color_array=token_map_max, 
                                token_concat='##',
                                skip_tokens=["**NULL**"],
                                color_map=self.color_map,
                                mapping_range=self.mapping_range,
                                )

        occ_sens_class_colorized, _ = colorize_tokens2(tokens=x_text, 
                                color_array=occ_sens_class, 
                                token_concat='##',
                                skip_tokens=["**NULL**"],
                                color_map='gray',
                                mapping_range=self.mapping_range,
                                )

        return colorized_text_mean, color_bar_mean, \
            colorized_text_max, color_bar_max, occ_sens_class_colorized, n, np.sum(np.abs(token_map), axis=0) #last=[1, num_classes] = "importance per class"
    

class OcclusionSensitivityImage:    #CG 27.02.2024
    def __init__(self, nn_module, patch_size, color_map='turbo', map_colors_to_min_max=True) -> None:
        self.model = nn_module
        self.color_map = color_map
        self.patch_size = patch_size
        if isinstance(self.patch_size, int):
            self.patch_size = [self.patch_size, self.patch_size]
        self.map_comors_to_min_max = map_colors_to_min_max
        self.mapping_range = [0.15,0.85]

    def __call__(self, input_ids, image):
        """
        input_ids is just forwarded to the model (beware to forward the correct input_ids depending on your model)
        image: assuming to be 2D image with shape [1, 1, H, W]
        """
        #compute outputs unmasked
        outputs = self.model(input_ids, image)
        outputs = outputs.detach().cpu().numpy()
        outputs_pred = [1 if x >= 0.5 else 0 for x in outputs[0]]

        #compute mean of image for replacement
        replace_token = image.mean().item()

        #compute output diffs with patches
        n = 0
        occ_sens_class = np.empty((0, outputs.shape[1]))
        token_map = np.empty((0, outputs.shape[1]))

        for i in range(0, image.shape[2], self.patch_size[0]):
            for j in range(0, image.shape[3], self.patch_size[1]):
                #replace patch with replace_token
                img_occ = image.clone()
                img_occ[:, :, i:i+self.patch_size[0], j:j+self.patch_size[1]] = replace_token
                outputs_masked = self.model(input_ids, img_occ)
                outputs_masked = outputs_masked.detach().cpu().numpy()
                #see, if classes changed. If yes, store 1, else 0
                outputs_masked_pred = [1 if x >= 0.5 else 0 for x in outputs_masked[0]]
                occ_sens_class_i = [1 if x != y else 0 for x, y in zip(outputs_pred, outputs_masked_pred)]
                if np.sum(occ_sens_class_i) > 0:
                    n += 1
                occ_sens_class_i = np.array([occ_sens_class_i])
                occ_sens_class = np.concatenate((occ_sens_class, occ_sens_class_i), axis=0)
                #compute difference
                diff = outputs - outputs_masked
                #store difference in token_map
                token_map = np.concatenate((token_map, diff), axis=0)
                #free memory
                del img_occ, outputs_masked, diff
        del outputs

        #token_map shape = [image.shape[2]//self.patch_size[0] * image.shape[3]//self.patch_size[1], num_classes]
        #as negative values are more important, we take invert the values
        token_map = -token_map
        #copy of token_map for visualization
        token_map_max = token_map.copy()
        token_map_mean = token_map.copy()
        #token_map is now of shape [image.shape[2]//self.patch_size[0] * image.shape[3]//self.patch_size[1], num_classes]
        token_map_mean = np.mean(token_map_mean, axis=1)
        token_map_max = np.max(token_map_max, axis=1)
        #see if classes changed. If yes, store 1, else 0
        occ_sens_class = np.sum(occ_sens_class, axis=1)
        occ_sens_class[occ_sens_class > 0] = 1

        #map to min max range:
        if self.map_comors_to_min_max:
            token_map_mean = (token_map_mean - np.min(token_map_mean)) / (np.max(token_map_mean) - np.min(token_map_mean)+1e-7)
            token_map_max = (token_map_max - np.min(token_map_max)) / (np.max(token_map_max) - np.min(token_map_max)+1e-7)
        
        assert len(token_map) == len(token_map_max) == len(token_map_mean) == len(occ_sens_class)

        colorized_image_mean = colorize_img(
                                [image.shape[2], image.shape[3]],
                                self.patch_size,
                                token_map_mean, 
                                self.color_map, 
                                mapping_range=self.mapping_range,
        )
        """
        colorized_image_max = colorize_img(
                                [image.shape[1], image.shape[2]],
                                self.patch_size,
                                token_map_max, 
                                self.color_map, 
                                mapping_range=self.mapping_range,
        )
        """
        occ_sens_class_colorized = colorize_img(
                                [image.shape[2], image.shape[3]], 
                                self.patch_size,
                                occ_sens_class, 
                                "gray", 
                                mapping_range=self.mapping_range,
        )

        return colorized_image_mean, occ_sens_class_colorized, n, np.sum(np.abs(token_map), axis=0) #last=[1, num_classes] = "importance per class"


class OcclusionSensitivityTabularData:    #CG 21.05.2024
    def __init__(self, nn_module, replace_token, skip_token="**NULL**", color_map='turbo', map_colors_to_min_max=True,
                 joined_occ_interval=[None,None]
                 ):
        """
        joined_occ_interval: the given interval will be occluded and handled as one data entry
        """
        self.model = nn_module
        self.color_map = color_map
        self.map_colors_to_min_max = map_colors_to_min_max
        self.mapping_range = [0.15,0.85] # [0.15,0.85] herein the colormap is not too dark for unimportant tokens and thus still readable
        self.replace_token = replace_token
        self.skip_token = skip_token
        self.joined_occ_entries = joined_occ_interval

    def __call__(self, x_tabular, x_tabular_plain, x_img,
                 ):
        """
        x_tabular is a vector of numbers
        x_tabular_plain contains the plain data entries
        x_img is the image
        """

        #compute outputs unmasked
        outputs = self.model(x_tabular, x_img)
        outputs = outputs.detach().cpu().numpy()
        outputs_pred = [1 if x >= 0.5 else 0 for x in outputs[0]]

        if self.replace_token is None: #take the mean of the syllables
            #cast x_tokens[0] to float
            x_tabular_float = x_tabular[0].float()
            #take mean and round
            replace_token = int(torch.mean(x_tabular_float).round())
        else:
            replace_token = self.replace_token

        n = 0 #number of data entries that forced a class change

        #tqdm, _ = optional_import("tqdm", name="tqdm")
        occ_sens_class = np.empty((0, outputs.shape[1]))
        tabular_map = np.empty((0, outputs.shape[1]))

        i = 0
        while i < len(x_tabular[0]):
            print("i", i)    #TODO: delete
            if i == self.joined_occ_entries[0]:
                replaced_data = x_tabular[0][i:self.joined_occ_entries[1]].clone()
                x_tabular[0][i:self.joined_occ_entries[1]] = replace_token*torch.ones(self.joined_occ_entries[1]-i, dtype=torch.long)
            else:
                #replace data entry with replace_token
                replaced_data = x_tabular[0][i].clone()
                x_tabular[0][i] = replace_token
            #compute outputs masked
            outputs_masked = self.model(x_tabular, x_img)
            outputs_masked = outputs_masked.detach().cpu().numpy()
            #see, if classes changed. If yes, store 1, else 0
            outputs_masked_pred = [1 if x >= 0.5 else 0 for x in outputs_masked[0]]
            occ_sens_class_i = [1 if x != y else 0 for x, y in zip(outputs_pred, outputs_masked_pred)]
            if np.sum(occ_sens_class_i) > 0:
                n += 1
            occ_sens_class_i = np.array([occ_sens_class_i])
            occ_sens_class = np.concatenate((occ_sens_class, occ_sens_class_i), axis=0)
            #compute difference
            diff = outputs - outputs_masked
            #store difference in tabular_map
            tabular_map = np.concatenate((tabular_map, diff), axis=0)
            #restore data entry
            if i == self.joined_occ_entries[0]:
                x_tabular[0][i:self.joined_occ_entries[1]] = replaced_data
                i = self.joined_occ_entries[1]
            else:
                x_tabular[0][i] = replaced_data
            i += 1
            #free memory
            del replaced_data, outputs_masked
        del outputs

        #as negative values are more important, we take the invers of the values
        tabular_map = -tabular_map
        #copy of tabular_map for visualization
        tabular_map_max = tabular_map.copy()
        tabular_map_mean = tabular_map.copy()
        #tabular_map is now of shape [len(x_tabular[0]), num_classes]
        #compute the mean of the difference for each data entry
        tabular_map_mean = np.mean(tabular_map_mean, axis=1)
        #compute the max of the difference for each data entry
        tabular_map_max = np.max(tabular_map_max, axis=1)
        #see if classes changed. If yes, store 1, else 0
        occ_sens_class = np.sum(occ_sens_class, axis=1)
        occ_sens_class[occ_sens_class > 0] = 1

        #map to min max range:
        if self.map_colors_to_min_max:
            tabular_map_mean = (tabular_map_mean - np.min(tabular_map_mean)) / (np.max(tabular_map_mean) - np.min(tabular_map_mean)+1e-7)
            tabular_map_max = (tabular_map_max - np.min(tabular_map_max)) / (np.max(tabular_map_max) - np.min(tabular_map_max)+1e-7)

        assert len(tabular_map) == len(tabular_map_max) == len(tabular_map_mean) == len(occ_sens_class)

        colorized_tabular_mean, color_bar_mean = colorize_tokens(
                                tokens=x_tabular_plain,
                                color_array=tabular_map_mean,
                                color_map=self.color_map,
                                mapping_range=self.mapping_range,
                                cbar_length = 50
        )
        colorized_tabular_max, color_bar_max = colorize_tokens(
                                tokens=x_tabular_plain,
                                color_array=tabular_map_max,
                                color_map=self.color_map,
                                mapping_range=self.mapping_range,
                                cbar_length = 50
        )
        occ_sens_class_colorized, _ = colorize_tokens(
                                tokens=x_tabular_plain,
                                color_array=occ_sens_class,
                                color_map='gray',
                                mapping_range=self.mapping_range,
        )

        return colorized_tabular_mean, color_bar_mean, colorized_tabular_max, color_bar_max, occ_sens_class_colorized, n, \
            np.sum(np.abs(tabular_map), axis=0), np.sum(np.abs(tabular_map), axis=1)
            #axis=0: [1, num_classes] = "importance per class"
            #axis=1: [1, len(x_tabular[0])] = "importance per data entry"