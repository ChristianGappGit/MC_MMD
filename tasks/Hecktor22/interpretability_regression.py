"""
Interpretability of regression model for the prediction of RFS time task (i.e. Task2 of hecktor22)

usage: python interpretability_regression.py run_path model_name
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")
import sys
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImage, Spacing, ResizeWithPadOrCrop, EnsureType,
)
import nibabel as nib
from config_hecktor22 import Config, load_config

from MultiResNetMLP_Regression import ResMLPRegression
from MultiVITMLP_Regression import ViTMLPRegression

# own imports for interpretability
from interpretability_hecktor22.occlusion_sensitivity import OcclusionSensitivityModality, OcclusionSensitivity


#BEWARE: copied from hecktor_task2_RFS_outcome_pred.py, modified version with train_img_path for backward compatibility
class MultimodalDataset(Dataset):
    def __init__(self, data, img_path, transform=None):
        self.data = data
        #check if path already in img name
        if self.data[0]["img_files"][0].startswith(img_path):
            self.img_path = '' #can be reset to empty string
        self.img_path = img_path
        self.preprocess = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the images
        img_files = self.data[idx]["img_files"]
        #print(img_files)
        for img_file in img_files:
            if 'CT' in img_file:
                CT_img = os.path.join(self.img_path,img_file)
                if self.preprocess is not None:
                    CT_img = self.preprocess(CT_img)
                else:
                    CT_img = nib.load(CT_img).get_fdata().astype(np.float32) 
                CT_img = torch.tensor(CT_img, dtype=torch.float32)
                #must return PT_img as well, herein we want to return None, but then batch is not subscriptable anymore, hence we return torch.tensor([0])
                PT_img = torch.tensor([0])
            elif 'PT' in img_file:
                PT_img = os.path.join(self.img_path, img_file)
                if self.preprocess is not None:
                    PT_img = self.preprocess(PT_img)
                else:
                    PT_img = nib.load(PT_img).get_fdata().astype(np.float32)
                PT_img = torch.tensor(PT_img, dtype=torch.float32)
                CT_img = torch.tensor([0])
            else: #throw error
                raise ValueError("Image is not CT or PET image")

        # Load the clinical information
        clinical_info = self.data[idx]["clinical_info"]
        #clinical_info = [clinical_info] #convertion to list was needed, but not anymore!
        #convert the clinical_info to a tensor
        clinical_info = torch.tensor(clinical_info, dtype=torch.float32)
        #print("clinical_info", clinical_info)
        # Load the labels
        relapse = torch.tensor([self.data[idx]["Relapse"]], dtype=torch.float32)
        rfs = torch.tensor([self.data[idx]["RFS"]], dtype=torch.float32)

        return {
            "CT_img": CT_img,
            "PT_img": PT_img,
            "clinical_info": clinical_info,
            "Relapse": relapse,
            "RFS": rfs,
            "img_files": img_files
        }

def main(run_path, model_name):
    if run_path is None:
        raise ValueError("run_path is None. Please provide a valid run_path.")
    if not os.path.exists(run_path):
        raise FileNotFoundError(f"run_path {run_path} does not exist.")
    
    #----------- Load Configuration File -----------#
    config_file = f"{run_path}/default_hecktor22.yaml"
    config = load_config(config_file)
    print(config)

    #paths hard coded:
    # Path to the dataset
    main_data_path = '/home/christian/data/Hecktor22'
    train_data_path = os.path.join(main_data_path, 'hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022')
    if config.data.use_segmented_imgs:
        train_img_path = os.path.join(train_data_path, 'imagesTr_segmented_cropped') #same shape, but voxels not in segment set to zero (= mask / (label!=0) * source_img)
    else:
        train_img_path = os.path.join(train_data_path, 'imagesTr') 
    

    # Define the model
    def Net(arch: str):
        print("Loading Regression model.")
        if arch == "ViTMLP":
            return ViTMLPRegression(
                in_channels=config.net.in_channels,
                img_size=config.net.img_size,
                patch_size=config.net.patch_size,
                spatial_dims=config.net.spatial_dims,
                num_classes=config.net.num_classes,
                num_clinical_features=config.net.num_clinical_features, #TODO: check if this is correct
                hidden_size_vision=config.net.hidden_size,
                mlp_dim = config.net.mlp_dim,
                num_heads = config.net.num_heads,
                num_vision_layers = config.net.num_vision_layers,
                dropout_rate=config.net.dropout_rate,
                qkv_bias=config.net.qkv_bias,
                pretrained_vision_net=config.net.pretrained_vision_net,
                model_path=None, #...
                act = config.net.act_ViT,
                only_vision=config.net.only_vision,
                only_clinical=config.net.only_clinical,
            )
        elif arch == "ResMLP":
            return ResMLPRegression(
            in_channels=config.net.in_channels,
            img_size=config.net.img_size,
            spatial_dims=config.net.spatial_dims,
            num_classes=config.net.num_classes,
            num_clinical_features=config.net.num_clinical_features, #TODO: check if this is correct
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
            raise ValueError(f"Architecture {arch} not implemented. Please choose from ['ViTMLP', 'ResMLP']")


    model = Net(config.net.arch)
    #moad pth file
    model_checkpoint = torch.load(os.path.join(run_path, model_name))
    if 'state_dict' in model_checkpoint:    # for newer trained models (with epoch information, and optimizer state)
        model.load_state_dict(model_checkpoint['state_dict'])
        print("model was trained with {} epochs.".format(model_checkpoint['epoch']))
    else:
        model.load_state_dict(model_checkpoint) #for backward compatibility
    model.to(config.device)
    print("Model loaded successfully.")
    print("model = ", model)

    #load data from npy files
    train_data = np.load(os.path.join(run_path, 'train_data.npy'), allow_pickle=True)
    val_data = np.load(os.path.join(run_path, 'val_data.npy'), allow_pickle=True)
    num_samples = config.occlusion.num_examples if config.occlusion.num_examples < len(val_data) else len(val_data)
    val_data = val_data[:num_samples]
    print("val_data loaded successfully.")
    print("len(val_data) = ", len(val_data))
    print("val_data 0 to 5 = ", val_data[:5])

    #INTERPRETABILTY:
    interpretability_path = os.path.join(run_path, 'interpretability', model_name)
    if not os.path.exists(interpretability_path):
        os.makedirs(interpretability_path)

    val_transforms = Compose([
            LoadImage(image_only=True, ensure_channel_first=True),
            Spacing(pixdim=config.spacing, mode=("bilinear")),
            ResizeWithPadOrCrop( #ensures constant spatial_size for all images, does central cropping
            spatial_size=config.net.img_size,
            method="symmetric", #padding
            ),
            EnsureType(),
    ])

    if config.data.cache:
        print("Not implemented for this task.")
    training_dataset = MultimodalDataset(train_data, train_img_path, val_transforms) #val_transforms and train_transforms are the same
    val_dataset = MultimodalDataset(val_data, train_img_path, val_transforms)
    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=False) #shuffle set to false for better analysis
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) #batch_size=1 for interpretability


    #Interpretability
    #model input is tabular, img
    #model output is RFS time in days

    #Occlusion Sensitivity Set Up
    occ_sens_modality = OcclusionSensitivityModality(model, maks_tabular_each=config.occlusion.tabular_mask, 
                                                     intervall_mask_together=[0,2],
                                                    mask_size_vision=config.occlusion.mask_size if config.occlusion.mask_patches else None, #None is whole image
                                                    )#hard coded here (first two entries in tabular data are the gender bits)

    """ clinical_info = {
    "Gender": M=[1,0], F=[0,1]
    "Age": int
    "Weight": int
    "Tobacco": 1 or 0
    "Alcohol": 1 or 0
    "Performance status": int
    "HPV status (0=-, 1=+)"
    "Surgery": 1 or 0
    "Chemotherapy": 1 or 0
    }
    """
    #--------------------Interpretability--------------------#
    #explanation of n_entries:
    # 2: 1 vision, 1 tabular (tabular all at once masked)
    # 10: 1 vision, 9 tabular (tabular masked every entry (gender has two bits)
    # 11: 1 vision, 10 tabular (tabular masked one bit by one bit) #not implemented as it is not useful
    n_entries = 1+config.net.num_clinical_features-1 if config.occlusion.tabular_mask else 2 # 1 (vision) + x tabular - 1 (gender has 2 bits)
    diff_probs = np.zeros((len(val_loader), n_entries, config.net.num_classes)) #shape = (num_samples, num_modalities, num_classes)
    modalitiy_contribution_per_item = np.zeros((len(val_loader), n_entries)) #shape = (num_samples, num_modalities, num_classes)
    image_names = []
    model.eval()
    from tqdm import tqdm
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            CT_img = batch['CT_img'].to(config.device) if len(batch['CT_img'].shape) > 1 else None
            PT_img = batch['PT_img'].to(config.device) if len(batch['PT_img'].shape) > 1 else None
            clinical_info = batch['clinical_info'].to(config.device)
            relapse = batch['Relapse'].to(config.device)
            rfs = batch['RFS'].to(config.device)
            indata = {'tabular': clinical_info, 'img': CT_img, }
            diff_probs_i, modality_contribution_per_item_i = occ_sens_modality(indata)
            diff_probs[i] = diff_probs_i
            modalitiy_contribution_per_item[i] = modality_contribution_per_item_i
            image_names.append(batch['img_files'])

    occ_sens_dict = [
        {
            'image_name' : image_name,
            'diff_probs': diffs,
            'modalitiy_contribution_per_item' : contribs,
        }
        for image_name, diffs, contribs in zip(image_names, diff_probs, modalitiy_contribution_per_item)
    ]

    modalitiy_contribution = np.mean(modalitiy_contribution_per_item, axis=0) #shape = (num_modalities)

    #write out to file
    occ_sens_file = f"{interpretability_path}/occ_sens_modality.txt"

    attribute_names = ["gender", "age", "weight", "tobacco", "alcohol", 
                       "performance status", "HPV status", "surgery", 
                       "chemotherapy", "vision"] if config.occlusion.tabular_mask else ["tabular", "vision"]
    with open(occ_sens_file, "w") as file:
        file.write("Occlusion Sensitivity Modality\n")
        file.write("Attribute Contribution\n")
        for attribute, contribution in zip(attribute_names, modalitiy_contribution):
            file.write(f"{attribute} : {contribution}\n")
        #write tabular sum (i.e. all tabular attributes masked at once)#
        tabular_sum = np.sum(modalitiy_contribution[np.where(np.array(attribute_names) != "vision")]) #sum of all tabular attributes
        file.write("Tabular Sum : {}\n\n".format(tabular_sum))
        if config.occlusion.tabular_mask: #additional write out of normalized values for each tabular attribute
            file.write("Attribute Contribution (normalized)\n")
            for attribute, contribution in zip(attribute_names, modalitiy_contribution):
                if attribute == "vision":
                    continue
                else:
                    file.write(f"{attribute} : {contribution/tabular_sum}\n")
        for item in occ_sens_dict:
            file.write(str(item) + "\n")

    #3D occ sens for vision
    occ_sens = OcclusionSensitivity(
        nn_module=model, mask_size=config.occlusion.mask_size, mode=config.occlusion.mode,
        n_batch=config.occlusion.n_batch, overlap=config.occlusion.overlap, 
        #stride=config.occlusion.stride, #stride is removed in newer monai versions (use overlap instead)
    )
    #...

if __name__ == "__main__":
    #sys.argv[1] = run_path
    #sys.argv[2] = model_name
    if len(sys.argv) != 3:
        raise ValueError("Please provide a run_path and model_name: usage: python interpretability_regression.py run_path model_name")
    main(sys.argv[1], sys.argv[2])