"""
function to evaluate the performance of a regression model
usage:
python3 performance_regression.py /path/to/run model_name
"""


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #TODO: change, if multiple GPU needed
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

from lifelines.utils import concordance_index


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
            "RFS": rfs
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
    print("val_data loaded successfully.")
    print("val_data 0 to 5 = ", val_data[:5])

    #PERFORMANCE:
    performance_path = os.path.join(run_path, 'performance', model_name)
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)

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
    train_loader = DataLoader(training_dataset, batch_size=config.train.batch_size, shuffle=False) #shuffle set to false for better analysis
    val_loader = DataLoader(val_dataset, batch_size=config.val.batch_size, shuffle=False)

    def create_metrics(loader, loader_name='val'):
        create_metrics.counter += 1
        targets = []
        predictions = []
        rfs_times = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                CT_img = batch['CT_img'].to(config.device) if len(batch['CT_img'].shape) > 1 else None
                PT_img = batch['PT_img'].to(config.device) if len(batch['PT_img'].shape) > 1 else None
                clinical_info = batch['clinical_info'].to(config.device)
                relapse = batch['Relapse'].to(config.device)
                rfs = batch['RFS'].to(config.device)
                img = CT_img
                output = model(clinical_info, img)
                predictions = np.concatenate([predictions, output.detach().cpu().numpy().flatten()])
                targets = np.concatenate([targets, relapse.detach().cpu().numpy().flatten()])
                rfs_times = np.concatenate([rfs_times, rfs.detach().cpu().numpy().flatten()])
    
        #concordance input to file (for later analysis)
        if create_metrics.counter == 1:
            with open(os.path.join(performance_path, 'concordance_input.txt'), 'w') as f:
                f.write(f"targets: {targets}\n")
                f.write(f"predictions: {predictions}\n")
                f.write(f"rfs_times: {rfs_times}\n")
        else:
            with open(os.path.join(performance_path, 'concordance_input.txt'), 'a') as f:
                f.write(f"targets: {targets}\n")
                f.write(f"predictions: {predictions}\n")
                f.write(f"rfs_times: {rfs_times}\n")
        #concordance index
        c_index = concordance_index(rfs_times, predictions, targets)
        print(f"{loader_name} concordance index: {c_index}")
        #print out to file
        if create_metrics.counter == 1:
            with open(os.path.join(performance_path, 'c_index.txt'), 'w') as f:
                f.write(f"{loader_name} concordance index: {c_index}\n")
        else:
            with open(os.path.join(performance_path, 'c_index.txt'), 'a') as f:
                f.write(f"{loader_name} concordance index: {c_index}\n")
    create_metrics.counter = 0

    create_metrics(train_loader, 'train')
    create_metrics(val_loader, 'val')


if __name__ == '__main__':
    run_path = sys.argv[1]
    model_name = sys.argv[2]
    main(run_path, model_name)
