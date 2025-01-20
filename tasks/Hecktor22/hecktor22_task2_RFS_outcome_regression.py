"""
HECKTOR 22 Grand Challenge
https://hecktor.grand-challenge.org/Overview/

Task 2: The prediction of patient outcomes, namely Recurrence-Free Survival (RFS) from the FDG-PET/CT images and available clinical data.
Images: 
    FDG-PET/CT images
Clinical data
    Gender,Age,Weight,Tobacco,Alcohol,Performance status,"HPV status (0=-, 1=+)",Surgery,Chemotherapy  //(missing modalities for some patients)

MODEL:
    Multimodal Transformer

Metric:
Concordance index (C-index)
Best of challenge: 0.682 (on private (not available) dataset)

Interpretability:
    done afterwards in order to get an importance metric for each modality
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")

# Importing Libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImage, Spacing, ResizeWithPadOrCrop, RandAffine,
    ScaleIntensity, Orientation, RandFlip, EnsureType,
    LoadImaged, Spacingd, ResizeWithPadOrCropd, RandAffined, RandFlipd, EnsureTyped,
    ToTensord,
)
import nibabel as nib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------------import models from BRSET--------------------------------------------------------
import sys
#insert path to BRSET
sys.path.insert(0, '/home/christian/projects/BRSET')
#print("sys path", sys.path)
from MultiResNetMLP import ResMLPNet #is in BRSET
from MultiVitMLP import ViTMLPNet #is in BRSET
# ----------------------------------------------------------------------------------------------------------------------------------------

# OLD IMPORTS
#from ViTMLPNet import ViTMLPNet

#local model import
from MultimodalTransformer import MultimodalTransformer

from MultiResNetMLP_Regression import ResMLPRegression
from MultiVITMLP_Regression import ViTMLPRegression

from config_hecktor22 import Config, load_config

#----------- Load Configuration File -----------#
config_file = "default_hecktor22.yaml"
config = load_config(config_file)
print(config)

# Path to the dataset
main_data_path = '/home/christian/data/Hecktor22'
train_data_path = os.path.join(main_data_path, 'hecktor2022_training_corrected_v3/hecktor2022_training/hecktor2022')
if config.data.use_segmented_imgs:
     train_img_path = os.path.join(train_data_path, 'imagesTr_segmented_cropped') #same shape, but voxels not in segment set to zero (= mask / (label!=0) * source_img)
else:
     train_img_path = os.path.join(train_data_path, 'imagesTr') 
train_img_labels_path = os.path.join(train_data_path, 'labelsTr')
train_clinical_info_file = os.path.join(train_data_path, 'hecktor2022_clinical_info_training.csv')
label_file = os.path.join(train_data_path, 'hecktor2022_endpoint_training.csv')
# Path to the output folder
output_path = '/home/christian/projects/Hecktor22/output' #not in data path, but here, in project path
if not os.path.exists(output_path):
    os.makedirs(output_path)

#----------- Create Directory for output data -----------#
x = 0
run_path = f'{output_path}/run_{x}'
while(os.path.isdir(run_path)):
    x = x + 1
    run_path = f'{output_path}/run_{x}'
os.mkdir(path=run_path)

#copy config_file
with open(config_file, 'r') as fp1, \
     open(f"{run_path}/{config_file}", 'w') as fp2:
    results = fp1.read()
    fp2.write(results)

#TensorBoard Setup
logdir = f'{run_path}/logs'
os.mkdir(path=logdir)
writer = SummaryWriter(log_dir=logdir)

# Load the clinical information
clinical_info = pd.read_csv(train_clinical_info_file)
"""
PatientID,Task 1,Task 2,CenterID,Gender,Age,Weight,Tobacco,Alcohol,Performance status,"HPV status (0=-, 1=+)",Surgery,Chemotherapy
"""
Task2 = clinical_info['Task 2']

#create a key value list of the patients that have Task 2, together with their clinical information
"""
{["PatientID", "Task 2", "CenterID" "Gender", "Age", "Weight", "Tobacco", "Alcohol", "Performance status", "HPV status (0=-, 1=+)", "Surgery", "Chemotherapy"]}
"""
patient_clinical_info = []
for i in range(len(Task2)):
    if Task2[i] == 1:
        patient_clinical_info.append([clinical_info['PatientID'][i], clinical_info['CenterID'][i], clinical_info['Gender'][i], clinical_info['Age'][i], clinical_info['Weight'][i], clinical_info['Tobacco'][i], clinical_info['Alcohol'][i], clinical_info['Performance status'][i], clinical_info['HPV status (0=-, 1=+)'][i], clinical_info['Surgery'][i], clinical_info['Chemotherapy'][i]])

# Load the labels
label_info = pd.read_csv(label_file)
"""
PatientID,Relapse,RFS
"""
PatientID = label_info['PatientID']
label_relapse = label_info['Relapse']
label_rfs = label_info['RFS']

#List images, if they end with .nii.gz
img_files = [f for f in os.listdir(train_img_path) if f.endswith('.nii.gz')]
if config.data.use_only_CT:
    img_files = [os.path.join(train_img_path, f) for f in img_files if 'CT' in f] #hence other non CT images removed.
else:
    img_files = [os.path.join(train_img_path, f) for f in img_files]
#sort the img_files
img_files.sort()

def get_mean_std(data, cols=[1,2], key="clinical_info"):
    """
    Calculate the mean and std of the clinical information for each column
    """
    train_mean = []
    train_std = []
    for col in cols: #age, weight only
        print([patient[key][col] for patient in data])
        col_mean = np.mean([patient[key][col] for patient in data if ~np.isnan(patient[key][col])])
        col_std = np.std([patient[key][col] for patient in data if ~np.isnan(patient[key][col])])
        print(col_mean, col_std)
        train_mean.append(col_mean)
        train_std.append(col_std)
    print(train_mean, train_std)
    return train_mean, train_std

def z_norm(data, mean, std, cols=[1,2], key="clinical_info"):
    """
    Normalize the clinical information
    """
    for patient in data:
        for i, col in enumerate(cols):
            patient[key][col] = (patient[key][col] - mean[i]) / std[i]
            #check if patient[key][col] is an array, then convert it to a list
            if isinstance(patient[key][col], np.ndarray):
                patient[key][col] = patient[key][col].tolist()
    return

def vectorized(clinical_info):
    """
    Vectorize the clinical information
    "Gender": M=[1,0], F=[0,1]
    "Age": int
    "Weight": int
    "Tobacco": 1 or 0
    "Alcohol": 1 or 0
    "Performance status": int
    "HPV status (0=-, 1=+)"
    "Surgery": 1 or 0
    "Chemotherapy": 1 or 0
    """
    #Gender
    if clinical_info[2] == 'M':
        clinical_info[2] = [1.0, 0.0]
    else:
        clinical_info[2] = [0.0, 1.0]
    #Age
    clinical_info[3] = [float(clinical_info[3])]
    #Weight
    clinical_info[4] = [float(clinical_info[4])]
    #Tobacco
    clinical_info[5] = [float(clinical_info[5])]
    #Alcohol
    clinical_info[6] = [float(clinical_info[6])]
    #Performance status
    clinical_info[7] = [float(clinical_info[7])]
    #HPV status
    clinical_info[8] = [float(clinical_info[8])]
    #Surgery
    clinical_info[9] = [float(clinical_info[9])]
    #Chemotherapy
    clinical_info[10] = [float(clinical_info[10])]
    #return the vectorized clinical information
    return clinical_info[2:] #exclude the patientID and the CenterID

#create a key value pair of each Patient with "PatientID", "img_file(s)", "clinical_info", "Relapse", "RFS",
patient_data = []
#attention: the order of the patients in the label_info is not the same as in the img_files, hence we need to match them
for i in range(len(PatientID)):
    img_indeces = [] #reset
    for j in range(len(img_files)):
        if PatientID[i] in img_files[j]:
            img_indeces.append(j)
            if len(img_indeces) == 2: #break if CT and, if necessary, PT image is found
                break
    if len(img_indeces) == 0: #if no image is found, skip the patient
        continue
    #check if the patient has Task 2
    #find the patient in the patient_clinical_info list and store its index
    patient_index = None
    for k in range(len(patient_clinical_info)):
        if PatientID[i] in patient_clinical_info[k]:
            patient_index = k
            break #break the loop, since we found the patient in the patient_clinical_info list
    if patient_index is None:
        continue # if the patient is not in the patient_clinical_info list, skip the patient
    #add the patient to the patient_data list
    #access the img_files with the img_indeces
    patient_data.append([PatientID[i], [img_files[index] for index in img_indeces] , vectorized(patient_clinical_info[patient_index]), label_relapse[i], label_rfs[i]])
    #repeat for all patients

#check if the patient_data list is correct
print(patient_data[0])

#now make key values of the patient_data list
#shuffle data, without shuffling the patients
np.random.seed(42)
np.random.shuffle(patient_data)
train_data_all = [
    {
        "PatientID": patient[0],
        "img_files": patient[1],
        "clinical_info": patient[2],
        "Relapse": patient[-2],
        "RFS": patient[-1]
    }
    for patient in patient_data
]

for i in range(10):
    print(train_data_all[i])

def mode(arr):
    """
    Calculate the mode of a list
    """
    if len(arr) == 0:
        return None
    #make sure not do modifie arr
    #check if entries in arr are of lists, then flatten the list
    #print(arr[0])
    #print datatype arr[0]
    #print(type(arr[0]))
    #print(type(arr[0][0]))
    #print(arr)
    if isinstance(arr[0], list):
        arr = [item for sublist in arr for item in sublist]
    #return the element with the highest occurrence
    #print(arr)
    mode = max(set(arr), key=arr.count)
    #print(mode)
    return mode


print("Training data vectorized:")
for i in range(10):
    print(train_data_all[i])

def replace_nan_entries(processed_data, reference_data):
    #replace "nan" entries in the clinical information with the mean of the column (age and weight), or the mode (tobacco, alcohol, performance status, HPV status, surgery, chemotherapy)
    mean_age = np.mean([p["clinical_info"][1] for p in reference_data if ~np.isnan(p["clinical_info"][1])])
    mean_weight = np.mean([p["clinical_info"][2] for p in reference_data if ~np.isnan(p["clinical_info"][2])])
    mode_tobacco = mode([p["clinical_info"][3] for p in reference_data if ~np.isnan(p["clinical_info"][3])])
    mode_alcohol = mode([p["clinical_info"][4] for p in reference_data if ~np.isnan(p["clinical_info"][4])])
    mode_performance_status = mode([p["clinical_info"][5] for p in reference_data if ~np.isnan(p["clinical_info"][5])])
    mode_HPV_status = mode([p["clinical_info"][6] for p in reference_data if ~np.isnan(p["clinical_info"][6])])
    mode_surgery = mode([p["clinical_info"][7] for p in reference_data if ~np.isnan(p["clinical_info"][7])])
    mode_chemotherapy = mode([p["clinical_info"][8] for p in reference_data if ~np.isnan(p["clinical_info"][8])])

    for patient in processed_data:
        for i, c_data in enumerate(patient["clinical_info"]):
            if i == 0: #skip the patientID
                continue
            if np.isnan(c_data):
                if i == 1: #age
                    patient["clinical_info"][i] = [mean_age]
                elif i == 2: #weight
                    patient["clinical_info"][i] = [mean_weight]
                elif i == 3: #tobacco (take mode)
                    patient["clinical_info"][i] = [mode_tobacco]
                elif i == 4: #alcohol (take mode)
                    patient["clinical_info"][i] = [mode_alcohol]
                elif i == 5: #performance status (take mode)
                    patient["clinical_info"][i] = [mode_performance_status]
                elif i == 6: #HPV status (take mode)
                    patient["clinical_info"][i] = [mode_HPV_status]
                elif i == 7: #surgery (take mode)
                    patient["clinical_info"][i] = [mode_surgery]
                elif i == 8: #chemotherapy (take mode)
                    patient["clinical_info"][i] = [mode_chemotherapy]
    return

#split the data into training and validation data
train_data = train_data_all[:int(0.8*len(train_data_all))]
val_data = train_data_all[int(0.8*len(train_data_all)):]

#print some information about data size
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

#1 calculate the mean and std of the clinical information of the training data (not modifie nan entries)
train_mean, train_std = get_mean_std(train_data)

#2 replace the nan entries in the clinical information with the mean or mode of the training data
replace_nan_entries(processed_data=train_data, reference_data=train_data)
replace_nan_entries(processed_data=val_data, reference_data=train_data) #with mean or mode of data in training data!!!

#3 z-norm the clinical information of the training data, now that the nan entries are replaced
if config.data.apply_z_norm:
    z_norm(train_data, train_mean, train_std)
    z_norm(val_data, train_mean, train_std) #Note that for the validation data the mean and std of the training data is used.

print("Training data z_normed age and weight:")
for i in range(10):
    print(train_data[i])

#bring clinical data to a single vector of numbers instead of vector of lists
for patient in train_data:
    patient["clinical_info"] = [item for sublist in patient["clinical_info"] for item in sublist]
for patient in val_data:
    patient["clinical_info"] = [item for sublist in patient["clinical_info"] for item in sublist]

print("Training data z_normed age and weight (single vector):")
for i in range(10):
    print(train_data[i])


#plot some statistics of the datasets:
#distribution of the relapse and RFS
def plot_target_statistics(data, name):
    relapse = [p["Relapse"] for p in data]
    rfs = [p["RFS"] for p in data]
    fig, ax = plt.subplots(1, 4)
    ax[0].bar(["0", "1"], [len([r for r in relapse if r == 0]), len([r for r in relapse if r == 1])])
    ax[0].set_title(f"Relapse in {name}")
    ax[1].hist(rfs, bins=20)
    ax[1].set_title(f"RFS {name}")
    #now plot RFS for relapse 0 and 1
    ax[2].hist([r for i, r in enumerate(rfs) if relapse[i] == 0], bins=20)
    ax[2].set_title(f"RFS {name} relapse 0")
    ax[3].hist([r for i, r in enumerate(rfs) if relapse[i] == 1], bins=20)
    ax[3].set_title(f"RFS {name} relapse 1")
    #strech each plot
    fig.set_size_inches(18, 6)
    plt.savefig(os.path.join(run_path, f'target_statistics_{name}.png'))
    plt.close()

plot_target_statistics(train_data, "train")
plot_target_statistics(val_data, "val")
plot_target_statistics(train_data+val_data, "all")

#gender distribution
gender_train = [p["clinical_info"][0:2] for p in train_data] #extract gender bits
fig, ax = plt.subplots(1, 2)
n_m_train = len([g for g in gender_train if g == [1.0, 0.0]])
n_f_train = len([g for g in gender_train if g == [0.0, 1.0]])
ax[0].bar(["M", "F"], [n_m_train, n_f_train])
ax[0].set_title(f"Train (n = {n_m_train + n_f_train}).")
gender_val = [p["clinical_info"][0:2] for p in val_data]
n_m_val = len([g for g in gender_val if g == [1.0, 0.0]])
n_f_val = len([g for g in gender_val if g == [0.0, 1.0]])
ax[1].bar(["M", "F"], [n_m_val, n_f_val])
ax[1].set_title(f"Validation (n = {n_m_val + n_f_val}).")
plt.savefig(os.path.join(run_path, 'gender_statistics_train_val.png'))
plt.close()
#another figure with gender distribution in all data
fig, ax = plt.subplots(1, 1)
ax.bar(["M", "F"], [n_m_train + n_m_val, n_f_train + n_f_val])
ax.set_title(f"All data (n = {n_m_train + n_f_train + n_m_val + n_f_val}).")
plt.savefig(os.path.join(run_path, 'gender_statistics_all.png'))
plt.close()

#done with the data preparation
# save out the data to train_data.npy and val_data.npy, together with some statistics to statistics.txt
np.save(os.path.join(run_path, 'train_data.npy'), train_data)
np.save(os.path.join(run_path, 'val_data.npy'), val_data)
with open(os.path.join(run_path, 'statistics.txt'), 'w') as f:
    f.write(f"Training data size: {len(train_data)}\n")
    f.write(f"Validation data size: {len(val_data)}\n")
    f.write(f"all data: {len(train_data)+len(val_data)}\n")

class MultimodalCacheDataset(CacheDataset):
    """
    CacheDataset for the multimodal dataset
    """
    def __init__(self, data, transform=None):
        super().__init__(data, transform=transform)
        self.data = data
        return
    
    def __getitem__(self, idx):
        cached_data = super().__getitem__(idx)
        # Load the images
        img = cached_data["img_files"] #already loaded and transformed
        img_name = self.data[idx]["img_files"] #order must have not changed within caching data
        #extract last part of the img_name after /
        if isinstance(img_name, list): #if list, then take the last part of the first element
            img_name = img_name[0].split("/")[-1]
        if 'CT' in img_name:
            CT_img = img
            PT_img = torch.tensor([0])
        elif 'PT' in img_name:
            CT_img = torch.tensor([0])
            PT_img = img
        else:
            raise ValueError("Image is not CT or PET image")
        #Load from cached data and transform to tensor
        clinical_info = torch.tensor(cached_data["clinical_info"], dtype=torch.float32)
        relapse = torch.tensor([cached_data["Relapse"]], dtype=torch.float32)
        rfs = torch.tensor([cached_data["RFS"]], dtype=torch.float32)

        return {
            "CT_img": CT_img,
            "PT_img": PT_img,
            "clinical_info": clinical_info,
            "Relapse": relapse,
            "RFS": rfs
        }


class MultimodalDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.preprocess = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the images
        img_files = self.data[idx]["img_files"]
        #print(img_files)
        for img_file in img_files:
            if 'CT' in img_file:
                CT_img = img_file
                if self.preprocess is not None:
                    CT_img = self.preprocess(CT_img)
                else:
                    CT_img = nib.load(CT_img).get_fdata().astype(np.float32) 
                CT_img = torch.tensor(CT_img, dtype=torch.float32)
                #must return PT_img as well, herein we want to return None, but then batch is not subscriptable anymore, hence we return torch.tensor([0])
                PT_img = torch.tensor([0])
            elif 'PT' in img_file:
                PT_img = img_file
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
    
train_transforms = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        Spacing(pixdim=config.spacing, mode=("bilinear")),
        ResizeWithPadOrCrop( #ensures constant spatial_size for all images, does central cropping
            spatial_size=config.net.img_size,
            method="symmetric", #padding
        ),
        RandFlip(spatial_axis=[0,1], prob=config.train.randFlipProp),
        RandAffine(
            mode=('bilinear'),
            prob=config.train.randAffineProp, spatial_size=config.net.img_size, #default is None, image_size used then
            rotate_range=config.train.randAffineRotateRange,
            scale_range=config.train.randAffineScaleRange,
        ),
        EnsureType(),
])
train_transforms_dict = Compose([
        LoadImaged(image_only=True, keys=['img_files'], ensure_channel_first=True),
        Spacingd(pixdim=config.spacing, keys=['img_files'], mode=("bilinear")),
        ResizeWithPadOrCropd( #ensures constant spatial_size for all images, does central cropping
            spatial_size=config.net.img_size,
            method="symmetric", #padding
            keys=['img_files'],
        ),
        RandFlipd(spatial_axis=[0,1], keys=['img_files'], prob=config.train.randFlipProp),
        RandAffined(
            mode=('bilinear'),
            prob=config.train.randAffineProp, spatial_size=config.net.img_size, #default is None, image_size used then
            rotate_range=config.train.randAffineRotateRange,
            scale_range=config.train.randAffineScaleRange,
            keys=['img_files'],
        ),
        EnsureTyped(keys=['img_files'],),
])

val_transforms = Compose([
        LoadImage(image_only=True, ensure_channel_first=True),
        Spacing(pixdim=config.spacing, mode=("bilinear")),
        ResizeWithPadOrCrop( #ensures constant spatial_size for all images, does central cropping
        spatial_size=config.net.img_size,
        method="symmetric", #padding
        ),
        EnsureType(),
])
val_transforms_dict = Compose([
        LoadImaged(image_only=True, keys=['img_files'], ensure_channel_first=True),
        Spacingd(pixdim=config.spacing, keys=['img_files'], mode=("bilinear")),
        ResizeWithPadOrCropd( #ensures constant spatial_size for all images, does central cropping
            spatial_size=config.net.img_size,
            method="symmetric", #padding
            keys=['img_files'],
        ),
        EnsureTyped(keys=['img_files']),
])


if config.data.cache:
    training_dataset = MultimodalCacheDataset(train_data, train_transforms_dict)
    val_dataset = MultimodalCacheDataset(val_data, val_transforms_dict)
else:
    training_dataset = MultimodalDataset(train_data, train_transforms)
    val_dataset = MultimodalDataset(val_data, val_transforms)
train_loader = DataLoader(training_dataset, batch_size=config.train.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.val.batch_size, shuffle=False)

print(len(train_loader), len(val_loader))

#plot and save some examples
for i in range(2):
    sample = training_dataset[i]
    CT_img = sample['CT_img']
    PT_img = sample['PT_img']
    print(len(CT_img.shape), PT_img.shape)
    if len(CT_img.shape) == 1 and len(PT_img.shape) == 1:
        raise ValueError("No image data found")
    elif len(CT_img.shape) > 1 and len(PT_img.shape) == 1:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(CT_img[0, :, :, CT_img.shape[3]//2], cmap='gray')
    elif len(PT_img.shape) > 1 and len(CT_img.shape) == 1:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(PT_img[0, :, :, PT_img.shape[3]//2], cmap='hot')
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(CT_img[0, :, :, CT_img.shape[3]//2], cmap='gray')
        ax[1].imshow(PT_img[0, :, :, PT_img.shape[3]//2], cmap='hot') 
    plt.savefig(os.path.join(run_path, f'example_{i}.png'))
    plt.close()


#done with the data preparation

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

model = Net(config.net.arch)
print(model)
#save the model architecture
with open(os.path.join(run_path, 'model_architecture.txt'), 'w') as f:
    f.write(str(model))

# Define the optimizer adamW
optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)

# Define the loss function for regression: MSE (mean squared error)
loss_function = torch.nn.MSELoss().to(config.device)

#training parameters
epochs = config.train.epochs
val_interval = config.val.interval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def auc(targets, prediction_score):
    """AU ROC"""
    return roc_auc_score(targets, prediction_score)


def validate(val_loader, model, loss_fn, device): #compute concordance index as well
    model.eval()
    val_loss = 0
    predictions = []
    targets = []
    times = []
    with torch.no_grad():
        for batch in val_loader:
            CT_img = batch['CT_img'].to(device) if len(batch['CT_img'].shape) > 1 else None
            PT_img = batch['PT_img'].to(device) if len(batch['PT_img'].shape) > 1 else None
            clinical_info = batch['clinical_info'].to(device)
            relapse = batch['Relapse'].to(device)
            rfs = batch['RFS'].to(device)
            #for img in (CT_img, PT_img): #forward pass for both images, then backward pass (per patient)
            img = CT_img
            output = model(clinical_info, img)
            loss = loss_fn(output, rfs)
            val_loss += loss.item()
            predictions.append(output.detach().cpu().numpy().flatten())
            targets.append(relapse.detach().cpu().numpy().flatten())
            times.append(rfs.detach().cpu().numpy().flatten())
        epoch_val_loss = val_loss/len(val_loader)
        writer.add_scalar('epoch_loss/val', epoch_val_loss, epoch)
        val_concordance_index = concordance_index(np.concatenate(times), np.concatenate(predictions), np.concatenate(targets))
        writer.add_scalar('c_index/val', val_concordance_index, epoch)
    return epoch_val_loss, val_concordance_index

def train(train_loader, model, loss_fn, optimizer, device):
    model.train()
    train_loss = 0
    predictions = []
    targets = []
    times = []
    for steps, batch in enumerate(train_loader):
        CT_img = batch['CT_img'].to(device) if len(batch['CT_img'].shape) > 1 else None
        PT_img = batch['PT_img'].to(device) if len(batch['PT_img'].shape) > 1 else None
        clinical_info = batch['clinical_info'].to(device)
        relapse = batch['Relapse'].to(device)
        rfs = batch['RFS'].to(device)
        optimizer.zero_grad()
        #for img in (CT_img, PT_img): #forward pass for both images, then backward pass (per patient)
        img = CT_img
        output = model(clinical_info, img)
        #print("Compute LOSS:")
        #print(f"output_shape = {output.shape}")
        #print(f"output = {output.flatten()[:]}")
        #print(f"rfs (label)_shape = {rfs.shape}")
        #print(f"rfs = {rfs.flatten()[:]}")
        loss = loss_fn(output, rfs)
        #print the loss, and not the tensor
        print(f"loss = {loss.item()}")
        loss.backward()
        optimizer.step()
        predictions.append(output.detach().cpu().numpy().flatten())
        targets.append(relapse.detach().cpu().numpy().flatten())
        times.append(rfs.detach().cpu().numpy().flatten())
        train_loss += loss.item()
        writer.add_scalar('step_loss/train', loss.item(), epoch*len(train_loader)+steps)
    epoch_loss = train_loss/len(train_loader)    
    writer.add_scalar('epoch_loss/train', epoch_loss, epoch)
    c_index = concordance_index(np.concatenate(times), np.concatenate(predictions), np.concatenate(targets))
    writer.add_scalar('c_index/train', c_index, epoch)
    return epoch_loss, c_index

# Training Loop
best_c_index = 0
best_loss = 1e10
for epoch in range(epochs):
    train_loss, train_c_index= train(train_loader, model, loss_function, optimizer, device)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Training c-index: {train_c_index}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if epoch % val_interval == 0:
        val_loss, val_c_index = validate(val_loader, model, loss_function, device)
        if val_c_index > best_c_index:
            best_c_index = val_c_index
            torch.save(checkpoint, os.path.join(run_path, 'model_best_c_index.pth')) #FIXME: eventually change to *.pt ?
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(run_path, 'model_best_loss.pth'))
        #store current val model
        torch.save(checkpoint, os.path.join(run_path, 'model_last_val.pth'))
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation c-index: {val_c_index}")
    #store current model
    torch.save(checkpoint, os.path.join(run_path, 'model_last.pth'))

#done with the training

#Some evaluation metrics
# not further necerssary, since we have the concordance index already computed