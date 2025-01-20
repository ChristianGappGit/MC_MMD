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

import torch
import torch.nn as nn
import numpy as np

from monai.data.meta_tensor import MetaTensor
from monai.networks.utils import eval_mode
from monai.transforms import Compose, GaussianSmooth, Lambda, ScaleIntensity, SpatialCrop
from monai.utils import deprecated_arg, ensure_tuple_rep, optional_import
from monai.visualize.visualizer import default_upsampler
import tqdm


class OcclusionSensitivity: #copied from monai, adapted to forward both image and tabular data
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


class OcclusionSensitivityModality:
    def __init__(self, model, maks_tabular_each=False, intervall_mask_together=None, mask_size_vision=None, replace_method = "zero"):
        self.model = model
        self.mask_tabular_each = maks_tabular_each
        self.intervall_mask_together = intervall_mask_together #for instance for Gender with one hot encoding (0,1) or (1,0) could be masked together
        #intervall_mask_together = (0,2) would mask the first two entries, i.e. (0,1). so beware that the last entry is not masked
        self.mask_size_vision = mask_size_vision #if None, mask_size will be whole image, else mask_size will be used (we recommend to use a big enough mask_size)
        self.replace_method = replace_method if replace_method in ["random", "zero"] else "random"

    def init_tensors_like(self, tensor):
        """
        function for initialization of tensors,
        return random or zero tensor with same shape and dtype and device as given tensor
        """
        if self.replace_method == "random":
            return torch.rand_like(tensor)
        elif self.replace_method == "zero":
            return torch.zeros_like(tensor)
        else:
            raise ValueError("replace_method must be 'random' or 'zero'.")

    
    def __call__(self, data):
        """
        data is dictionary with keys as modalities and values as tensors
        """
        #TODO: implement class changes, and random, zero replacement of tensors
        #extract modalities from data
        modalities = list(data.keys())
        print("modalities = ", modalities)
        #extact values
        values = list(data.values())

        tabular_token_len = 0
        if self.mask_tabular_each:
            for i in range(len(values)):
                if "tabular" in modalities[i]:
                    tabular_token_len += values[i].shape[1]
                    if self.intervall_mask_together is not None:
                        tabular_token_len -= (self.intervall_mask_together[1] - self.intervall_mask_together[0] - 1)
        print("tabular_token_len = ", tabular_token_len)

        #get the output of the model
        #we have to forward the values through the model, not as list
        #check if data is already on same device as model
        if next(self.model.parameters()).device != values[0].device:
            values = [v.to(next(self.model.parameters()).device) for v in values]

        print("values", values)
        output = self.model(*values)
        print("output_unmasked", output)

        #now we mask each modality and get the output
        #we will return the difference in output
        n_masked_entries = len(modalities)+tabular_token_len-1 if self.mask_tabular_each else len(modalities)
        print("n_masked_entries = ", n_masked_entries)
        differences_probs_per_modality_per_class = np.zeros((n_masked_entries, output.shape[1]))
        i = 0 #index for modalities + extra
        shift = 0
        while i < len(modalities):
            masked_values = values.copy()
            masked_values[i] = values[i].clone()

            #TABULAR
            if self.mask_tabular_each and "tabular" in modalities[i]:
                #mask each entry of tabular data extra
                print("masked_values[i]_shape = ", masked_values[i].shape)
                print("values[i]_shape = ", values[i].shape)
                differences_probs_per_token_tabular = np.zeros((tabular_token_len, output.shape[1]))
                j = 0 # index for tabular data access
                k = 0 # for storage of differences (max_k = max_j - self.intervall_mask_together[1] - self.intervall_mask_together[0])
                #here implement logic for masking specific intervall in tabular data at once 
                while j < (values[i].shape[1]):
                    print("j = ", j)
                    masked_values = values.copy() #reset
                    masked_values[i] = values[i].clone()
                    if self.intervall_mask_together is not None:
                        if j == self.intervall_mask_together[0]:
                            masked_values[i][:,j:self.intervall_mask_together[1]] = self.init_tensors_like(values[i][:,j:self.intervall_mask_together[1]])
                            j = self.intervall_mask_together[1] - 1 #skip intervall
                        else:
                            masked_values[i][:,j] = self.init_tensors_like(values[i][:,j])
                    else:
                        masked_values[i][:,j] = self.init_tensors_like(values[i][:,j])
                    print("j = ", j)
                    print("masked_values = ", masked_values)
                    masked_output = self.model(*masked_values)
                    #get the difference
                    diff = output - masked_output
                    diff = diff.cpu().detach().numpy()
                    print("diff_shape = ", diff.shape)
                    print("difference_probs_per_token_tabular_shape = ", differences_probs_per_token_tabular.shape)
                    print("difference_probs_per_token_tabular[k]_shape = ", differences_probs_per_token_tabular[k].shape)
                    differences_probs_per_token_tabular[k] = diff
                    j += 1
                    k += 1
                print("differences_probs_per_token_tabular_shape = ", differences_probs_per_token_tabular.shape)
                print("differences_probs_per_modality_per_class = ", differences_probs_per_modality_per_class.shape)
                shift += tabular_token_len
                differences_probs_per_modality_per_class[i:i+shift] = differences_probs_per_token_tabular

            #VISION
            elif self.mask_size_vision is not None and modalities[i] in ["vision", "image", "img"]:
                img_size = values[i].shape[2:]
                if len(img_size) != len(self.mask_size_vision):
                    raise ValueError("mask_size_vision must have same dimension as image size. Image size = ", img_size, "mask_size_vision = ", self.mask_size_vision)
                for m, p in zip(img_size, self.mask_size_vision):
                    if m < p:
                        raise ValueError("mask_size should be smaller than img_size.")
                    if m % p != 0:
                        raise ValueError("img_size should be divisible by mask_size for occlusion.")
                print("img_size = ", img_size)
                #mask the vision data
                patch_size = [ip / ms for ip, ms in zip(img_size, self.mask_size_vision)]
                n_masked_patches = int(np.prod(patch_size))
                if len(patch_size) == 2:
                    nx, ny = patch_size
                    #mask the modality
                    differences_probs_per_vision_patch = np.zeros((n_masked_patches, output.shape[1]))
                    mask = self.init_tensors_like(values[i][:, :, :self.mask_size_vision[0], :self.mask_size_vision[1]])
                    #iterate over nx, ny
                    for j in range(nx):
                        for k in range(ny):
                            masked_values = values.copy() #reset
                            masked_values[i] = values[i].clone()
                            masked_values[i][:, :, j:j+self.mask_size_vision[0], k:k+self.mask_size_vision[1]] = mask
                            print("masked_values = ", masked_values)
                            masked_output = self.model(*masked_values)
                            #get the difference
                            diff = output - masked_output
                            diff = diff.cpu().detach().numpy()
                            differences_probs_per_vision_patch[j*ny+k] = diff
                else:
                    nx, ny, nz = patch_size #3D
                    #ensure integer
                    nx, ny, nz = int(nx), int(ny), int(nz)
                    #mask the modality
                    differences_probs_per_vision_patch = np.zeros((n_masked_patches, output.shape[1]))
                    mask = self.init_tensors_like(values[i][:, :, :self.mask_size_vision[0], :self.mask_size_vision[1], :self.mask_size_vision[2]])
                    #iterate over nx, ny, nz
                    for j in range(nx): 
                        for k in range(ny): 
                            for l in range(nz):
                                masked_values = values.copy() #reset
                                masked_values[i] = values[i].clone()
                                masked_values[i][:, :, j:j+self.mask_size_vision[0], k:k+self.mask_size_vision[1], l:l+self.mask_size_vision[2]] = mask
                                print("masked_values = ", masked_values)
                                masked_output = self.model(*masked_values)
                                #get the difference
                                diff = output - masked_output
                                diff = diff.cpu().detach().numpy()
                                differences_probs_per_vision_patch[j*ny*nz+k*nz+l] = diff
                if shift != 0:
                    differences_probs_per_modality_per_class[i+shift-1] = np.sum(differences_probs_per_vision_patch, axis=0)
                else:
                    differences_probs_per_modality_per_class[i] = np.sum(differences_probs_per_vision_patch, axis=0) 

            #Whole modality (TABULAR, VISION, ...)
            else:   #don't care about modality, just mask all the attributes
                #mask the modality
                print("values[i] = ", values[i])
                print("i = ", i)
                masked_values[i] = self.init_tensors_like(values[i]) #TODO: check, maybe better use random numbers here?
                print("masked_values = ", masked_values)
                masked_output = self.model(*masked_values)
                #get the difference
                diff = output - masked_output
                diff = diff.cpu().detach().numpy()
                if shift != 0:
                    differences_probs_per_modality_per_class[i+shift-1] = diff
                else:
                    differences_probs_per_modality_per_class[i] = diff
            #update i
            i += 1

        #compute modality contribution by mapping to range [0,1]
        #differences_probs_per_modality_per_class_shape = (n_masked_entries, n_classes)
        #modality_contribution_per_modality_per_class_shape = (n_masked_entries, 1)
        #map to range [0,1]
        modality_contribution_per_modality_per_class = np.abs(differences_probs_per_modality_per_class) #absolut values
        modality_contribution_per_modality_per_class = np.sum(modality_contribution_per_modality_per_class, axis=1) #sum of differences over classes (i.e. dynamic)
        sum_diff = np.sum(modality_contribution_per_modality_per_class)
        modality_contribution_per_modality_per_class = modality_contribution_per_modality_per_class / sum_diff

        print("Occlusion Sensitivity DONE.")
        return differences_probs_per_modality_per_class, modality_contribution_per_modality_per_class