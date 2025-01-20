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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #TODO: change, if multiple GPU needed
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
    with open(f"{perform_dir}/train_data_processed.txt", "r") as f:
        train_data = f.readlines()
        #ensure value key pair, should be like the following:
        #{'img': 'img16224.jpg', 'tab': [-1.2908358897205403, 0.412535160779953, 0.7479466795921326, 0.797730028629303, 0.3335990905761719, 1.0759280920028687, 1.2494597434997559, -0.6464053392410278, 0.5923640727996826, -0.5181387066841125, 0.15065860748291016, -1.3793858289718628, -0.7832409143447876, -0.05242030695080757, 0.5579866170883179, 2.195936918258667, -1.0902981758117676, 0.9378358721733093, -0.12805257737636566, 0.8463577032089233, -0.9732387065887451, -0.4635332524776459, 0.28359872102737427, -1.2197680473327637, -0.09388529509305954, -0.5406694412231445, 0.1827174872159958, 0.2954365313053131, -0.2947392463684082, 0.3206520974636078, -0.1287541687488556, 0.3193542327841861, 1.0, 1.0, 1.0, 1.0], 'targets': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        train_data = [eval(entry) for entry in train_data]
    with open(f"{perform_dir}/val_data_processed.txt", "r") as f:
        val_data = f.readlines()
        val_data = [eval(entry) for entry in val_data]

    train_transforms = transforms.Compose(
            [
                transforms.Resize(config.net.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
    )

    val_transforms = transforms.Compose(
                [
                transforms.Resize(config.net.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
    )


    train_params = {
        "batch_size": 1, #config.train.batch_size, #overwritten
        "shuffle": config.train.shuffle,
        "num_workers": config.train.num_workers,
        "pin_memory": True,
    }

    val_params = {
        "batch_size": 1, #config.val.batch_size, #overwritten
        "shuffle": config.val.shuffle,
        "num_workers": config.val.num_workers,
        "pin_memory": True,
    }

    train_dataset = MultimodalDataset(data=train_data, img_path=img_path, transform=train_transforms)
    train_loader = DataLoader(train_dataset, **train_params)

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
    
    #interpretability setup
    for name, _ in model.named_modules(): print(name)
   
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

        title = img_name
        titles.append(title)
        log_scales.append(False)
        text_to_file = f"{title}\n"
        d_targets_int = [int(i) for i in targets[0]]
        text_to_file += f"GT:    {d_targets_int}"
        text_to_file += f"\nPred:  {pred_targets}\nProbs: {['{:0.4f}'.format(entry) for entry in pred_probs]}\n"
        #write out text file with text_to_file (done at the end of this function)

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
        
        temp_text = ""
        if os.path.exists(f"{output_dir}/interpretability_information.txt") and num > 0:
            with open(f"{output_dir}/interpretability_information.txt", 'r') as fp:
                temp_text = fp.read()
        with open(f"{output_dir}/interpretability_information.txt", "w") as f:
            f.write(temp_text)
            if temp_text != "": f.write("\n")
            f.write(text_to_file)

        #return ims, titles, log_scales, [file_cam, file_cbar_cam], [file_occ_sens, file_cbar_occ_sens], importance_per_class_tabular, importance_per_token_tabular
        return ims, titles, log_scales, [None, None], [file_occ_sens, file_cbar_occ_sens], importance_per_class_tabular, importance_per_token_tabular


    #now iterate over ?loader and compute saliency for each item
    #--------------------------------------------------------------------------------
    num_examples = config.occlusion.num_examples #number of examples to be processed
    for row, d in enumerate(val_loader):   #val_loader, training_loader
        print("\nProcessing item...", row)
        print("d[image].shape", d["image"].shape)
        ims, titles, log_scales, _, colorized_data_occ_sens, \
        importance_tabular, importance_per_token_tabular = saliency(model, d, row)

        if row == 0:
            importance_per_class_tabular_all = np.empty((0,importance_tabular.shape[0]))
            importance_per_token_tabular_all = np.empty((0,importance_per_token_tabular.shape[0]))

        importance_per_class_tabular_all = np.concatenate((importance_per_class_tabular_all, [importance_tabular]), axis=0)
        importance_per_token_tabular_all = np.concatenate((importance_per_token_tabular_all, [importance_per_token_tabular]), axis=0)

        if row == (num_examples-1):
            break
    
    #OCC sensitivity
    #compute mean importance per class
    importance_per_class_tabular_mean = np.mean(importance_per_class_tabular_all, axis=0)
    importance_tabular_mean = np.mean(importance_per_class_tabular_mean)
    importance_per_class_tabular_mean_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_tabular_mean])

    importance_per_token_tabular_mean = np.mean(importance_per_token_tabular_all, axis=0)
    #map to 0 1 range
    importance_per_token_tabular_mean = (importance_per_token_tabular_mean / np.sum(importance_per_token_tabular_mean))


    #save mean importance per class to file
    with open(f"{output_dir}/interpretability_information.txt", "a") as f:
        f.write(f"\n\nOcc_sens:\n")
        f.write(f"Mean importance per class (tabular):            [{importance_per_class_tabular_mean_str}]\n")
        f.write("\nImportance per token (tabular):\n")
        f.write(", ".join([f"{entry:0.2f}" for entry in importance_per_token_tabular_mean]))
        
    print("Finished with Interpretability ...")
    
    #postprocessing done in python script postprocessing.py
    #-----------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------- in this code unused --------------------------------------------------------------------------------------#
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