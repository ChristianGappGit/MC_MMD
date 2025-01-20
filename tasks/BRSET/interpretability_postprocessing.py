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

Usage: python3 interpretability_postprocessing.py output/run_x/

"""
#----------- Import dependencies -----------#

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")

import torch
import sys
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
from MultiResNetMLP import ResMLPNet
from MultiDenseNetMLP import DenseMLPNet
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
    perform_dir = perform_dir[:-1] if perform_dir[-1] == "/" else perform_dir
    model_dir = f"{perform_dir}" #must exist
    model_name = "model_best_auc.pth" #must exist
    output_dir = f"{perform_dir}/interpretability" #can exist, otherwise created
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    #----------- Load Configuration File -----------#
    config_file = f"{perform_dir}/default.yaml"
    config = load_config(config_file)
    print(config)

    #Setup folders:
    datadir = "/home/christian/data/BRSET"
    img_path = f"{datadir}/images"
    label_path = f"{datadir}/labels"

    #----------- Load Data -----------#
    #read in the processed data
    #ensure value key pair, should be like the following:
    #{'img': 'img16224.jpg', 'tab': [-1.2908358897205403, 0.412535160779953, 0.7479466795921326, 0.797730028629303, 
    # 0.3335990905761719, 1.0759280920028687, 1.2494597434997559, -0.6464053392410278, 0.5923640727996826, -0.5181387066841125, 
    # 0.15065860748291016, -1.3793858289718628, -0.7832409143447876, -0.05242030695080757, 0.5579866170883179, 2.195936918258667, 
    # -1.0902981758117676, 0.9378358721733093, -0.12805257737636566, 0.8463577032089233, -0.9732387065887451, -0.4635332524776459, 
    # 0.28359872102737427, -1.2197680473327637, -0.09388529509305954, -0.5406694412231445, 0.1827174872159958, 0.2954365313053131, 
    # -0.2947392463684082, 0.3206520974636078, -0.1287541687488556, 0.3193542327841861, 1.0, 1.0, 1.0, 1.0], 
    # 'targets': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    with open(f"{perform_dir}/val_data_processed.txt", "r") as f:
        val_data = f.readlines()
        val_data = [eval(entry) for entry in val_data]

    val_transforms = transforms.Compose(
                [
                transforms.Resize(config.net.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
    )

    val_params = {
        "batch_size": 1, #config.val.batch_size, #overwritten
        "shuffle": False, #config.val.shuffle, #overwritten
        "num_workers": config.val.num_workers,
        "pin_memory": True,
    }

    val_dataset = MultimodalDataset(data=val_data, img_path=img_path, transform=val_transforms)
    val_loader = DataLoader(val_dataset, **val_params)

    #print some info about val_data and train data shapes
    print("\nval_data[0]", val_data[0])

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
                    use_pretrained_vit=config.net.pretrained_vision_net,
                    only_vision=config.net.only_vision,
                    only_clinical=config.net.only_clinical,
                )
            elif name == "DenseMLP":
                return DenseMLPNet(
                    in_channels=config.net.in_channels,
                    img_size=config.net.img_size,
                    spatial_dims=config.net.spatial_dims,
                    num_classes=config.net.num_classes,
                    num_clinical_features=config.net.num_clinical_features+config.data.embedding_dim_comorbidities-1, #solved
                    dropout_rate=config.net.dropout_rate,
                    pretrained_vision_net=config.net.pretrained_vision_net,
                    act = config.net.act,
                    only_vision=config.net.only_vision,
                    only_clinical=config.net.only_clinical,
                )
            elif name == "ResMLP":
                return ResMLPNet(
                    in_channels=config.net.in_channels,
                    img_size=config.net.img_size,
                    spatial_dims=config.net.spatial_dims,
                    num_classes=config.net.num_classes,
                    num_clinical_features=config.net.num_clinical_features+config.data.embedding_dim_comorbidities-1, #solved
                    dropout_rate=config.net.dropout_rate,
                    conv1_t_size=config.net.conv1_t_size,
                    conv1_t_stride=config.net.conv1_t_stride,
                    pretrained_vision_net=config.net.pretrained_vision_net,
                    model_path=None, #...
                    act = config.net.act,
                    only_vision=config.net.only_vision,
                    only_clinical=config.net.only_clinical,
                )
            else:
                raise ValueError(f"Architecture {name} not found")

    model = Net(config.net.name)
    print(model)

    # Load the model
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=config.device)["state_dict"],strict=False)
    print(f"Model loaded from {model_dir}/{model_name}")
    model.to(config.device)
    model.eval()

    #num_patches computation:
    num_patches_height = config.net.img_size[0] // config.net.patch_size[0]
    num_patches_width = config.net.img_size[1] // config.net.patch_size[1]

    def reshape_transform(tensor, height=num_patches_height, width=num_patches_width):
        #print("in_shape: ", tensor.shape)
        result = tensor[: , 1: , :].reshape(tensor.size(0), #.. "result = tensor[: , 1: , :]"1:" because first channel is the class token TODO: ok?
            height, width, tensor.size(-1)) #-1 should stay the same as in the input tensor
        #Bring the channels to the first dimension,
        #like in CNNs
        #print(result.shape)
        result = result.transpose(2,3).transpose(1,2)
        #print("out_shape: ", result.shape)
        return result

    reshape_function = reshape_transform if config.net.name == "ViTMLP" else None
    
    #interpretability setup
    for name, _ in model.named_modules(): print(name)
    #select a proper layer for gradcam
    if config.net.name == "ViTMLP":
        target_layers_vision = [model.vision_model[0].blocks[-1].norm1] #choose any layer before the final attention block, because it must be a list (even if only one layer) in order to be iterable
    else: #DenseMLP or ResMLP
        target_layers_vision = [model.vision_model[0].layer4[-1].bn2] #= model.vision_model.0.layer4.2.bn2

    print("target_layers_vision", target_layers_vision)

    #target_layers_tab_data = ...

    gradcam = GradCAM(           #only works for 2D images
            model=model, target_layers=target_layers_vision, 
            use_cuda = True, reshape_transform=reshape_function,
    )

    occ_sens = OcclusionSensitivity(
        nn_module=model, mask_size=config.occlusion.mask_size, mode=config.occlusion.mode,
        n_batch=config.occlusion.n_batch, overlap=config.occlusion.overlap, 
        #stride=config.occlusion.stride, #stride is removed in newer monai versions (use overlap instead)
    )
    occ_sens_img = OcclusionSensitivityImage(
        nn_module=model, patch_size=config.occlusion.mask_size, color_map='turbo',
        map_colors_to_min_max=True
    )
    occ_sens_tab_data = OcclusionSensitivityTabularData( #attention on what to set for replace_token! set to very high number as no high numbers in tabular data
        nn_module=model, replace_token=1e3, skip_token=None, color_map='turbo', map_colors_to_min_max=True,
        joined_occ_interval = [1,config.data.embedding_dim_comorbidities] #solved
    )


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
        input_data_plain = d["tabular_plain"]
        targets = d["targets"].to(config.device)
        img_name = d["img_name"][0]
        img_name = img_name[:-4] if img_name[-4] == "." else img_name #remove file ending
        pred_logits = net(input_tabular, img)
        pred_probs = pred_logits.detach().cpu() #already probality values, as sigmoid is applied at the end of the model
        pred_probs = pred_probs[0]  #dereferenced once
        pred_targets = get_targets_from_probs(pred_probs)

        #reshape img ?
        print("img.shape before reshaping",img.shape)
        img_plot = img #[:,0,:,:] #TODO: check if some adaptation needed here...
        print("img.shape after reshaping",img_plot.shape)
        ims.append(img_plot)
        print("d[img_name]", d["img_name"])
        print("d[img_name][0]", d["img_name"][0])
        title = img_name
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
        #map to 0...1
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
        #print("\nStart Grad CAM on tabular data ...")

        #GradCam on tabular data
        #TODO: implement
        print("\nStart Occlusion Sensitivity on tabular data...")
        #Occ. sens. on tabular data
        occ_sens_tabular_mean, occ_sens_tabular_mean_bar, occ_sens_tabular_max, \
        _, occ_sens_tabular_class, n_changes_tabular, importance_per_class_tabular, importance_per_token_tabular = occ_sens_tab_data(input_tabular, input_data_plain, img)

        file_occ_sens = f'{output_dir}/{img_name}/{img_name}_OccSens_tabular.html'
        if not os.path.isdir(f"{output_dir}/{img_name}"):
            os.mkdir(f"{output_dir}/{img_name}")
        with open(file_occ_sens, 'w') as f:
            f.write("MEAN<br>")
            f.write(occ_sens_tabular_mean)
            #in html style: add two breaks
            f.write("<br><br>MAX<br>")
            f.write(occ_sens_tabular_max)
            f.write(f"<br><br>{n_changes_tabular} entries forced model to change the classification output. <br>")
            f.write(occ_sens_tabular_class)
        #save the color bar as html file
        if num == 0:
            file_cbar_occ_sens = f'{output_dir}/OccSens_tabular_cbar.html'
            with open(file_cbar_occ_sens, 'w') as f:
                f.write(occ_sens_tabular_mean_bar)
        else: file_cbar_occ_sens = None
        print(f"\nOCC sens on tabular data DONE.\nWrote {file_occ_sens}")

        #IMPORTANCE:
        #Occ_sens:
        importance_sum = importance_per_class_vision + importance_per_class_tabular
        importance_per_class_vision = importance_per_class_vision / importance_sum
        importance_per_class_tabular = importance_per_class_tabular / importance_sum

        #cam.. skip for now #TODO: implement

        #add to file
        #Occ sens
        importance_per_class_vision_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_vision])
        importance_per_class_tabular_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_tabular])
        text_to_file += f"Occ_sens:\n"
        text_to_file += f"Importance per class (tabular):            [{importance_per_class_tabular_str}]\n"
        text_to_file += f"Importance per class (vision):          [{importance_per_class_vision_str}]\n"
        text_to_file += f"Importance mean ratio (tabular : vision) = {np.round(np.mean(importance_per_class_tabular),2)} : {np.round(np.mean(importance_per_class_vision),2)}\n"

        #cam...
        #...
        
        temp_text = ""
        if os.path.exists(f"{output_dir}/interpretability_information.txt") and num > 0:
            with open(f"{output_dir}/interpretability_information.txt", 'r') as fp:
                temp_text = fp.read()
        with open(f"{output_dir}/interpretability_information.txt", "w") as f:
            f.write(temp_text)
            if temp_text != "": f.write("\n")
            f.write(text_to_file)

        #return ims, titles, log_scales, [file_cam, file_cbar_cam], [file_occ_sens, file_cbar_occ_sens], importance_per_class_tabular, importance_per_class_vision, cam_importance_text, cam_importance_vision
        return ims, titles, log_scales, [None, None], [file_occ_sens, file_cbar_occ_sens], importance_per_class_tabular, importance_per_class_vision, None, cam_importance_vision, importance_per_token_tabular


    def add_im(im, title, log_scale, row, col, num_examples, cmap, background=None, alpha=1.0,):
        #create new figure for one row and store to file
        #note that axes_row is asumed to be one dimensional here (as only one row is plotted)
        ax = axes[row, col] if num_examples > 1 else axes[col]
        print("im.shape", im.shape)
        im = im[0] if im.shape[0] == 1 else im
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu()
        print("im.shape[0] detached", im.shape)
        #img could be 3 channels
        if im.shape[0] == 3:
            im = im.permute(1,2,0)
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
            #background could be 3 channels
            if background.shape[0] == 3:
                background = background.permute(1,2,0)
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
    num_examples = config.occlusion.num_examples #number of examples to be processed
    for row, d in enumerate(val_loader):   #val_loader, training_loader
        print("\nProcessing item...", row)
        print("d[image].shape", d["image"].shape)
        ims, titles, log_scales, colorized_data_cam, colorized_data_occ_sens, \
        importance_tabular, importance_vision, cam_importance_tabular, cam_importance_vision, importance_per_token_tabular = saliency(model, d, row)
        #plot the results, save to file
        if row < config.occlusion.num_examples_to_plot:
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
                os.mkdir(f"{output_dir}/{titles[0]}") #should be already created in saliency function()
            plt.savefig(f"{output_dir}/{titles[0]}/{titles[0]}_interpretability_vision.png")
            plt.savefig(f"{output_dir}/{titles[0]}/{titles[0]}_interpretability_vision.pdf")
            plt.close()

        if row == 0:
            importance_per_class_tabular_all = np.empty((0,importance_tabular.shape[0]))
            importance_per_class_vision_all = np.empty((0,importance_vision.shape[0]))
            cam_importance_tabular_all = np.empty(0)
            cam_importance_vision_all = np.empty(0)
            importance_per_token_tabular_all = np.empty((0,importance_per_token_tabular.shape[0]))

        importance_per_class_tabular_all = np.concatenate((importance_per_class_tabular_all, [importance_tabular]), axis=0)
        importance_per_class_vision_all = np.concatenate((importance_per_class_vision_all, [importance_vision]), axis=0)
        cam_importance_tabular_all = np.concatenate((cam_importance_tabular_all, [cam_importance_tabular]), axis=0)
        cam_importance_vision_all = np.concatenate((cam_importance_vision_all, [cam_importance_vision]), axis=0)
        importance_per_token_tabular_all = np.concatenate((importance_per_token_tabular_all, [importance_per_token_tabular]), axis=0)

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

    #tabular importance per token
    importance_per_token_tabular_mean = np.mean(importance_per_token_tabular_all, axis=0)
    #map to 0 1 range is wrong, as we want to have each token's contribution and not just mapped to [0,1]
    #importance_per_token_tabular_mean = (importance_per_token_tabular_mean - importance_per_token_tabular_mean.min()) / (importance_per_token_tabular_mean.max() - importance_per_token_tabular_mean.min()+1e-6)
    tabular_sum = np.sum(importance_per_token_tabular_mean)
    importance_per_token_tabular_mean = importance_per_token_tabular_mean / tabular_sum #hence we have computed the contribution

    #CAM
    cam_tabular_mean = 0 # np.mean(cam_importance_tabular_all, axis=0) #TODO: change when implemented
    cam_vision_mean = np.mean(cam_importance_vision_all, axis=0)
    cam_sum = cam_tabular_mean + cam_vision_mean
    cam_tabular_mean = cam_tabular_mean / cam_sum
    cam_vision_mean = cam_vision_mean / cam_sum

    #save mean importance per class to file
    with open(f"{output_dir}/interpretability_information.txt", "a") as f:
        f.write(f"\n\nOcc_sens:\n")
        f.write(f"Mean importance per class (tabular):            [{importance_per_class_tabular_mean_str}]\n")
        f.write(f"Mean importance per class (vision):          [{importance_per_class_vision_mean_str}]\n")
        f.write(f"Mean importance ratio (tabular : vision) = {np.round(importance_tabular_mean,3)} : {np.round(importance_vision_mean,3)}\n")
        f.write("\nCAM:\n")
        f.write(f"Mean importance ratio (tabular : vision) = {np.round(cam_tabular_mean,3)} : {np.round(cam_vision_mean,3)}\n")
        f.write("\nImportance per token (tabular):\n")
        f.write(", ".join([f"{entry:0.3f}" for entry in importance_per_token_tabular_mean]))
        #now write relative to all modalites
        f.write("\nImportance per token (tabular) relative to all modalities:\n")
        f.write(", ".join([f"{entry:0.3f}" for entry in importance_per_token_tabular_mean*importance_tabular_mean])) #mp_i^l * m_i = m_i^l
        
    print("Finished with Interpretability ...")
    
    #postprocessing done in python script postprocessing.py
    #-----------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    main(sys.argv[1])