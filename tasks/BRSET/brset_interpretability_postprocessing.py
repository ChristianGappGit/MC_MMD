"""
A Brazilian Multilabel Ophthalmological Dataset (BRSET)
downloaded data from https://physionet.org/content/brazilian-ophthalmological/1.0.0/
github: https://github.com/luisnakayama/BRSET/tree/main

Data:
    2D images
    tabular data
Task:
    Multilabel Classification

Input: 
    - images, tabular data
    - labels

Output:
    -metric for importance of modaility
    -saliency maps
"""
#----------- Import dependencies -----------#

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")

import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from PIL import Image
from enum import Enum
#from monai.transforms import ResizeWithPadOrCrop, LoadImage, EnsureType
from torchvision import transforms
from monai.config import print_config
from monai.utils import set_determinism
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, f1_score
)
from MultiVitMLP import ViTMLPNet
from config import Config, load_config
from dataset_bar_plot import get_targets, create_bar_graph


#----------- Print Configurations  -----------#
torch.backends.cudnn.benchmark = True
print_config()

# own imports for interpretability
from interpretability.occlusion_sensitivity import OcclusionSensitivity, OcclusionSensitivityImage, OcclusionSensitivityTabularData
from interpretability.cam import GradCAM

from brset_multimodal_classification import MultimodalDataset, Diseases


def main(perform_dir: str):

    #----------- Print Configurations  -----------#
    torch.backends.cudnn.benchmark = True
    print_config()
    #perform_dir = sys.argv[1] #argutment passed to main() function
    model_dir = f"{perform_dir}" #must exist
    model_name = "model.pth" #must exist
    output_dir = f"{perform_dir}/interpretability" #can exist, otherwise created
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    #----------- Load Configuration File -----------#
    config_file = f"{perform_dir}/default_llama.yaml"
    config = load_config(config_file)
    print(config)

    #Setup folders:
    datadir = "/home/christian/data/BRSET"
    img_path = f"{datadir}/images"
    label_path = f"{datadir}/labels"

    #----------- Load Data -----------#
    #read in the processed data
    with open(f"{perform_dir}/train_data_processed.txt", "r") as f:
        train_data = f.readlines()
    with open(f"{perform_dir}/val_data_processed.txt", "r") as f:
        val_data = f.readlines()

    train_transforms = transforms.Compose(
            [
                transforms.Resize(config.net.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
    )

    val_transforms = transforms.Compose(
                [
                    transforms.Resize(config.net.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
    )

    train_params = {
        "batch_size": config.train.batch_size,
        "shuffle": config.train.shuffle,
        "num_workers": config.train.num_workers,
        "pin_memory": True,
    }

    val_params = {
        "batch_size": config.val.batch_size,
        "shuffle": config.val.shuffle,
        "num_workers": config.val.num_workers,
        "pin_memory": True,
    }

    train_dataset = MultimodalDataset(data=train_data, img_path=img_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, **train_params)

    val_dataset = MultimodalDataset(data=val_data, img_path=img_path, transform=val_transforms)
    val_loader = DataLoader(val_dataset, **val_params)

    #----------- Load Model -----------#
    # Define the model
    def Net(name: str):
        assert config.net.num_classes == len(Diseases), "Number of classes should be equal to the number of diseases"
        if name == None:
            raise ValueError("No architecture specified")
        elif name == "ViTMLP":
            return ViTMLPNet(
                in_channels=config.net.in_channels,
                img_size=config.net.img_size,
                patch_size=config.net.patch_size,
                spatial_dims=config.net.spatial_dims,
                num_classes=config.net.num_classes,
                num_clinical_features=config.net.num_clinical_features+config.data.embedding_dim_comorbidities-1, #solved
                hidden_size_vision=config.net.hidden_size,
                mlp_dim=config.net.mlp_dim,
                num_heads=config.net.num_heads,
                num_vision_layers=config.net.num_vision_layers,
                dropout_rate=config.net.dropout_rate,
                qkv_bias=config.net.qkv_bias,
                use_pretrained_vit=config.net.use_pretrained_vit
            )
        else:
            raise ValueError(f"Architecture {name} not found")

    model = Net(config.net.name)
    print(model)

    # Load the model
    model.load_state_dict(torch.load(f"{model_dir}/{model_name}"))
    print(f"Model loaded from {model_dir}/{model_name}")
    model.eval()

    #num_patches computation:
    num_patches_height = config.net.img_size[0] // config.net.patch_size[0]
    num_patches_width = config.net.img_size[1] // config.net.patch_size[1]

    def reshape_transform(tensor, height=num_patches_height, width=num_patches_width):
        print("in_shape: ", tensor.shape)
        result = tensor[: , : , :].reshape(tensor.size(0), #.. was "result = tensor[: , 1: , :]"1:" because first channel is the class token TODO: ok?
            height, width, -1) #-1 should stay the same as in the input tensor
        #Bring the channels to the first dimension,
        #like in CNNs
        print(result.shape)
        result = result.transpose(2,3).transpose(1,2)
        print("out_shape: ", result.shape)
        return result
    
    #interpretability setup
    for name, _ in model.named_modules(): print(name)
    #select a proper layer for gradcam
    target_layers_vision = [model.base_model.model.multimodal.vision_net.blocks[-1].norm1] #TODO: choose any layer before the final attention block, because it must be a list (even if only one layer) in order to be iterable
    #target_layers_tab_data = ...

    gradcam = GradCAM(           #only works for 2D images
            model=model, target_layers=target_layers_vision, 
            use_cuda = True, reshape_transform=reshape_transform,
    )
    occ_sens = OcclusionSensitivity(
        nn_module=model, mask_size=config.net.patch_size, mode=config.occlusion.mode,
        n_batch=config.occlusion.n_batch, overlap=config.occlusion.overlap, 
        #stride=config.occlusion.stride, #stride is removed in newer monai versions (use overlap instead)
    )
    occ_sens_img = OcclusionSensitivityImage(
        nn_module=model, patch_size=config.net.patch_size, color_map='turbo',
        map_colors_to_min_max=True
    )
    #occ_sens_tab_data = OcclusionSensitivityTabularData(


    def get_targets_from_probs(probabilities):
        #print("probabilities.shape", probabilities.shape)
        #print("p in probabilities[0]", probabilities[0])
        targets = [1 if (p >= 0.5) else 0 for p in probabilities]
        return targets
    
    def saliency(net, d, num):
        torch.set_printoptions(linewidth=200)
        ims = []
        titles = []
        log_scales = []

        #model computation of output probs
        img = d["image"].to(config.device)
        input_tabular = d["tabular"].to(config.device)
        targets = d["targets"].to(config.device)
        img_name = d["img_name"]
        pred_logits = net(input_tabular, img)
        pred_probs = pred_logits.detach().cpu() #already probality values, as sigmoid is applied at the end of the model
        pred_probs = pred_probs[0]  #dereferenced once
        pred_targets = get_targets_from_probs(pred_probs)

        #reshape img ?
        print("img.shape before reshaping",img.shape)
        img_plot = img #[:,0,:,:] #TODO: check if some adaptation needed here...
        print("img.shape after reshaping",img_plot.shape)
        ims.append(img_plot)
        print("d[name]", d["name"])
        print("d[name][0]", d["name"][0])
        title = img_name[0]
        titles.append(title)
        log_scales.append(False)
        text_to_file = f"{title}\n"
        d_targets_int = [int(i) for i in targets[0]]
        text_to_file += f"GT:    {d_targets_int}"
        text_to_file += f"\nPred:  {pred_targets}\nProbs: {['{:0.4f}'.format(entry) for entry in pred_probs]}\n"
        #write out text file with text_to_file (done at the end of this function)
        
        # GradCAM on image
        #targets set to None -- but OK as map for the highest scoring category is returned
        print("\n\n\n")
        print(input_tabular)
        res_cam, cam_importance_vision = gradcam(input_tensor_text=input_tabular, input_tensor_vision=img, targets=None) #if target_category = None -> base_cam.py computes this value itself
        res_cam = 1 - res_cam #invert the color map. (for turbo: red is important, blue is not)
        ims.append(res_cam)
        titles.append("GradCAM")
        log_scales.append(False)

        ims.append(res_cam)
        titles.append("GradCAM img") #for transparent plot
        log_scales.append(False)

        print("Grad CAM DONE.")
        print("\nStart Occlusion Sensitivity ...")

        # Occlusion sensitivity images
        occ_map, occ_most_prob = occ_sens(x=img, x_text=input_tabular)
        pred_label = np.argmax(pred_targets) #FIXME: problem with multiple class
        print("pred_label", pred_label)
        #= occ_sens_shape torch.Size([1, 14, y, x]), where 14 = num_classes
        #occ_map = occ_map.mean(dim=0, keepdim=True) #mean over all 14 (i.e. num_classes) channels
        #map to 0...1, and invert (for color interpretation)
        occ_map = occ_map[0,pred_label]
        occ_map = (occ_map - occ_map.min()) / (occ_map.max() - occ_map.min())
        occ_map = 1 - occ_map
        #Note that the color bar for the selected map is now in range [0,1]
        #if more occ_maps want to be selected and analyzed, the range must be adpted map per map
        ims.append(occ_map)
        titles.append("Occ. sens. (MONAI)")
        log_scales.append(False)

        occ_map2, occ_sens_class, n, importance_per_class_vision = occ_sens_img(input_tabular, img)
        ims.append(occ_map2)
        titles.append("Occ. sens. (CG)")
        log_scales.append(False)

        ims.append(occ_sens_class)
        titles.append("Occ. sens. class (CG), n=" + str(n)) #n: number of patches that forced model to change the classification output
        log_scales.append(False)

        print("\nOcclusion Sensitivity DONE.")
        print("\nStart Grad CAM on text ...")

        #GradCam on tabular data
        #TODO: implement

        #Occ. sens. on tabular data
        #TODO: implement



        #TODO: continue here... 


        #return ims, titles, log_scales, [file_cam, file_cbar_cam], [file_occ_sens, file_cbar_occ_sens], importance_per_class_text, importance_per_class_vision, cam_importance_text, cam_importance_vision
        return ims, titles, log_scales, [None, None], [None, None], None, importance_per_class_vision, None, cam_importance_vision


    def add_im(im, title, log_scale, row, col, num_examples, cmap, background=None, alpha=1.0,):
        #create new figure for one row and store to file
        #note that axes_row is asumed to be one dimensional here (as only one row is plotted)
        ax = axes[row, col] if num_examples > 1 else axes[col]
        print("im.shape", im.shape)
        im = im[0] if im.shape[0] == 1 else im
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu()
        print("im.shape[0] detached", im.shape)
        im_show = ax.imshow(im, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=25)
        ax.axis("off")
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        if col == 0: 
            fig.colorbar(im_show, ax=ax, cax=cax, ticks=np.arange(1e3,1e4,10)) #limits out of range, hence no numbers... FIXME: map to 0-255
        else: 
            fig.colorbar(im_show, ax=ax, cax=cax)
        if background is not None:
            background = background[0]  #background must be a tensor of shape [1,xdim,ydim]
            if isinstance(background, torch.Tensor):
                background = background.detach().cpu()
            ax.imshow(background, cmap="gray", alpha=0.7)
        return

    def add_row(ims, titles, log_scales, axes, row, num_examples):
        #Note that colorized_data is unused here
        for col, (im, title, log_scale) in enumerate(zip(ims, titles, log_scales)):
            cmap = "gray" if col == 0 else "turbo" #"viridis"
            if log_scale and im.min() < 0:
                im -= im.min()
            if col == 2:#col num to be changed as needed
                alpha = 0.9
                background = ims[0]
            else:
                alpha = 1.0
                background = None
            add_im(im, title, log_scale, row, col, num_examples, cmap, background=background, alpha=alpha)
        return

    #now iterate over ?loader and compute saliency for each item
    #--------------------------------------------------------------------------------
    num_examples = 500 #number of examples to be plotted
    for row, d in enumerate(val_loader):   #val_loader, training_loader
        print("\nProcessing item...", row)
        print("d[images].shape", d["images"].shape)
        ims, titles, log_scales, colorized_data_cam, colorized_data_occ_sens, \
        importance_tabular, importance_vision, cam_importance_tabular, cam_importance_vision = saliency(model, d, row)
        num_cols = len(ims)
        num_rows = 1
        subplot_shape = [num_rows, num_cols]
        figsize = [i * 5 for i in subplot_shape][::-1]
        #print("figsize", figsize)
        #gridspec = {'width_ratios': [1]*num_cols, 'height_ratios': [1]*num_examples}
        fig, axes = plt.subplots(*subplot_shape, figsize=figsize, facecolor="white",) #gridspec_kw=gridspec
        add_row(ims, titles, log_scales, axes, row=0, num_examples=1)
        #save figure
        plt.tight_layout()
        print("Saving Ouput of Interpretability process ...")
        #makes sure out_img_dir exists else create
        if not os.path.isdir(f"{output_dir}/{titles[0]}"):
            os.mkdir(f"{output_dir}/{titles[0]}")
        plt.savefig(f"{output_dir}/{titles[0]}/{titles[0]}_interpretability_vision.png")
        plt.savefig(f"{output_dir}/{titles[0]}/{titles[0]}_interpretability_vision.pdf")
        plt.close()

        if row == 0:
            importance_per_class_tabular_all = np.empty((0,importance_tabular.shape[0]))
            importance_per_class_vision_all = np.empty((0,importance_vision.shape[0]))
            cam_importance_tabular_all = np.empty(0)
            cam_importance_vision_all = np.empty(0)

        importance_per_class_tabular_all = np.concatenate((importance_per_class_tabular_all, [importance_tabular]), axis=0)
        importance_per_class_vision_all = np.concatenate((importance_per_class_vision_all, [importance_vision]), axis=0)
        cam_importance_tabular_all = np.concatenate((cam_importance_tabular_all, [cam_importance_tabular]), axis=0)
        cam_importance_vision_all = np.concatenate((cam_importance_vision_all, [cam_importance_vision]), axis=0)

        if row == (num_examples-1):
            break
    
    #OCC sensitivity
    #compute mean importance per class
    importance_per_class_tabular_mean = np.mean(importance_per_class_tabular_all, axis=0)
    importance_per_class_vision_mean = np.mean(importance_per_class_vision_all, axis=0)
    importance_tabular_mean = np.mean(importance_per_class_tabular_mean)
    importance_vision_mean = np.mean(importance_per_class_vision_mean)
    importance_sum = importance_tabular_mean + importance_vision_mean
    importance_tabular_mean = importance_tabular_mean / importance_sum
    importance_vision_mean = importance_vision_mean / importance_sum
    importance_per_class_tabular_mean_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_tabular_mean])
    importance_per_class_vision_mean_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_vision_mean])

    #CAM
    cam_tabular_mean = np.mean(cam_importance_tabular_all, axis=0)
    cam_vision_mean = np.mean(cam_importance_vision_all, axis=0)
    cam_sum = cam_tabular_mean + cam_vision_mean
    cam_tabular_mean = cam_tabular_mean / cam_sum
    cam_vision_mean = cam_vision_mean / cam_sum

    #save mean importance per class to file
    with open(f"{output_dir}/interpretability_information.txt", "a") as f:
        f.write(f"\n\nOcc_sens:\n")
        f.write(f"Mean importance per class (tabular):            [{importance_per_class_tabular_mean_str}]\n")
        f.write(f"Mean importance per class (vision):          [{importance_per_class_vision_mean_str}]\n")
        f.write(f"Mean importance ratio (tabular : vision) = {np.round(importance_tabular_mean,2)} : {np.round(importance_vision_mean,2)}\n")
        f.write("\nCAM:\n")
        f.write(f"Mean importance ratio (tabular : vision) = {np.round(cam_tabular_mean,2)} : {np.round(cam_vision_mean,2)}\n")
        
    print("Finished with Interpretability ...")
    
    #postprocessing done in python script postprocessing.py
    #-----------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------Later in the code (unused)--------------------------------------------------------------------------------------#
#from github (https://github.com/luisnakayama/BRSET/tree/main)
# Generate Saliency Maps
def get_saliency_map(model, input_image):
    model.eval()
    input_image.requires_grad_()
    output = model(input_image)
    max_idx = output.argmax()
    output[0, max_idx].backward()
    saliency_map, _ = torch.max(input_image.grad.data.abs(),dim=1)
    #saliency_map = input_image.grad.data.abs().max(1)[0]
    return saliency_map

#Shapley values
#similar to Occ sens. maps, but more computation time
#in order to get the importance of one single token, we need to compute the model output for all possible combinations of tokens
#complexity for one token: O(2^n). n = number of tokens
#in comparison to Occ sens: O(1)


if __name__ == "__main__":
    main(sys.argv[1])