"""
A Brazilian Multilabel Ophthalmological Dataset (BRSET)
downloaded data from https://physionet.org/content/brazilian-ophthalmological/1.0.0/
github: https://github.com/luisnakayama/BRSET/tree/main

Data:
    2D images
    tabular data
Task:
    Multilabel Classification
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from PIL import Image
from enum import Enum
#from monai.transforms import ResizeWithPadOrCrop, LoadImage, EnsureType
from torchvision.transforms import v2
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
from MultiSwinViTMLP import SwinViTMLPNet
from MultiSwinViTMLP_torch import SwinViTMLPNet_torch
from config import Config, load_config
from dataset_bar_plot import get_targets, create_bar_graph

class Diseases(Enum):
    """
    Enum class for the diseases
    """
    diabetic_retinopathy =      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    macular_edema =             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scar =                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nevus =                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    amd =                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vascular_occlusion =        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    hypertensive_retinopathy =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    drusens =                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    hemorrhage =                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    retinal_detachment =        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    myopic_fundus =             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    increased_cup_disc =        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    other =                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    No_Finding =                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   #additional class"

#TODO: complete this class
class MultimodalDataset(Dataset):
    def __init__(self, data, img_path, transform=None):
        self.data = data
        self.img_path = img_path
        self.transform = transform

    #for interpretability: get targets of img_name
    def get_targets(self, img_name):
        #return targets of img_name, where self.img_name contains the img_name
        self_img_name = np.array([d["img"] for d in self.data])
        print("img_name", img_name)
        print("self_img_name", self_img_name)
        if img_name not in self_img_name:
            print("img_name not in self_img_name")
            exit()
        index = np.where(self_img_name == img_name)[0][0]
        print(index)
        return np.array(self.data[index]["targets"])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data[idx]["img"]
        tab = torch.tensor(self.data[idx]["tab"], dtype=torch.float32)
        tab_plain_data = self.data[idx]["tab_plain"]
        targets = torch.tensor(self.data[idx]["targets"], dtype=torch.float32)
        #open jpeg file
        img_full_name = os.path.join(self.img_path, img_name)
        img = Image.open(img_full_name)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)
            
        #print("tab", tab)
        #print("targets", targets)

        return {
            "image": img,
            "img_name": img_name,
            "tabular": tab,
            "tabular_plain": tab_plain_data,
            "targets": targets
        }
    
def main():

    #----------- Print Configurations  -----------#
    torch.backends.cudnn.benchmark = True
    print_config()

    #----------- Load Configuration File -----------#
    config_file = './default.yaml'
    config = load_config(config_file)
    print(config)

    #----------- Set deterministic training for reproducibility  -----------#
    if config.seed >= 0:
        print(f"Setting random seed to {config.seed}")
        set_determinism(seed=None)#config.seed) #TODO: beware what to set here... !
    else:
        print("Random seed not set")

    #Setup folders:
    datadir = "/home/christian/data/BRSET"
    img_path = f"{datadir}/images"
    label_path = f"{datadir}/labels"

    outdir = "./output"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    xrun = 0
    run_path = f'{outdir}/run_{xrun}'
    while(os.path.isdir(run_path)):
        xrun = xrun + 1
        run_path = f'{outdir}/run_{xrun}'
    os.mkdir(path=run_path)

    #TensorBoard Setup
    logdir = f'{run_path}/logs'
    os.mkdir(path=logdir)
    writer = SummaryWriter(log_dir=logdir)

    #copy config_file
    with open(config_file, 'r') as fp1, \
        open(f"{run_path}/{config_file}", 'w') as fp2:
        results = fp1.read()
        fp2.write(results)

    def get_mean_std(data, cols=[0], key="tab"):
        """
        Calculate the mean and std of the clinical information for each column
        """
        train_mean = []
        train_std = []
        for col in cols:
            print([patient[key][col] for patient in data])
            col_mean = np.mean([patient[key][col] for patient in data if ~np.isnan(patient[key][col])])
            col_std = np.std([patient[key][col] for patient in data if ~np.isnan(patient[key][col])])
            #print(col_mean, col_std)
            train_mean.append(col_mean)
            train_std.append(col_std)
        print(train_mean, train_std)
        return train_mean, train_std

    def z_norm(data, mean, std, cols=[0], key="tab"):
        """
        Normalize the clinical information
        """
        assert len(mean) == len(std) == len(cols), "Mean, std and cols should have the same length"
        if mean is None or std is None: #if mean and std are not provided, calculate them
            mean, std = get_mean_std(data, cols, key)
        for patient in data:
            for i, col in enumerate(cols):
                patient[key][col] = (patient[key][col] - mean[i]) / std[i]
                #check if patient[key][col] is an array, then convert it to a list
                if isinstance(patient[key][col], np.ndarray):
                    patient[key][col] = patient[key][col].tolist()
        return mean, std

    #Load data
    #List images, if they end with .jpg
    img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
    #Load labels and tabular data from labels.csv file
    label_file = pd.read_csv(f'{label_path}/labels.csv')
    #print header now
    print(label_file.head())

    #label_file has image_id, matching the image file name
    #tabular data: age, comorbidities, diabetes time, insuline, patient_sex, exam_eye, diabetes, nationality (removed)
    comorbidities = [
        "NA", "diabetes1", "SAH", "diabetes", "hypothyroidism", "hydroxychloroquine", "hypercholesterolemia", "rheumatoid arthritis", "lupus", "vasculitis", 
        "hydroxichloroquine", "hydrocephalus", "cardiopathy", "cone dystrophy", "parkinson", "arrhythmia", "epilepsia", "dyslipidemia", "valvulopathy", "hepatitis c", 
        "meningioma", "sarcoidosis", "encephalic vascular accident", "hipoitireoidismo", "breast cancer", "chron disease", "psoriasis", "acute myocardium infarct", 
        "asthma", "cardiac insufficiency", "aneurysm", "talassemia", "sickle cell anemia", "mccune albright", "kidney transplant", "cerebral palsy", "arthrosis", 
        "catheterism", "vitiligo", "pulmonary embolism", "prostatic hyperplasia", "arthritis", "obesity", "hyperthyroidism", "sjogren", "tabagism", "chronic kidney disease", 
        "brain tumor", "human immunodeficiency virus", "multiple sclerosis", "herpetic encephalitis", "fibromyalgia", "alzheimer", "anemia", "hepatic transplant", 
        "chronic obstructive pulmonary disease", "lymphoma", "neurofibromatosis", "intracranial hypertension", "migraine", "osteoporosis", "chloroquine", "lung cancer", 
        "rhinitis", "thalassemia", "chagas", "syphilis", "deep vascular thrombosis", "behcet", "devic", "hipocolesterolemia", "dialysis", "chagas disease", "epilepsy", 
        "prolactinoma", "down syndrome", "hypertriglyceridemia", "graves disease", "trombose", "leucemia", "hypophysis adenoma", "juvenile arthritis", "adrenal hypoplasia", 
        "hashimoto disease", "albinism", "cerebrovascular accident", "parkison", "intestinal cancer", "thyroiditis", "ankylosing spondylitis", "cirrhosis", "policitemia vera",
        "multiple sclerosiss", "ulcerative colitis", "muscular dystrophy", "DPOC", "hepatic cancer"
        ] #len = 97

    def convertTensorToList(Tensor):
        """
        Convert a tensor to a list like:
        [a,b] -> [b] using Permute and AdaptiveAvgPool1d and Squeeze
        """
        #print("Tensor_shape", Tensor.shape) #is [len(comorbidities) = 97, embedding_dim_comorbidities = 30]
        #transform from [97, 30] to [30] with AdaptiveAvgPool1d
        Tensor = Tensor.permute(1, 0) #[30, 97]
        Tensor = torch.nn.AdaptiveAvgPool1d(1)(Tensor) #[30, 1]
        Tensor = Tensor.squeeze(-1) #[30]
        List = Tensor.cpu().detach().numpy().tolist()
        return List


    def tabular_data_str_one_hot(items):
        """
        Embedding the tabular data
        items: list of strings
        """
        #first: map "0" or "nan" to "NA"
        #print("comorbidities", comorbidities)
        items = ["NA" if c in ["0", "nan"] else c for c in items]
        #write comorbities to file, if items are not in comorbidities
        for c in items:
            if c not in comorbidities:  #FIXME: "NA" is in comorbidities, but not cannot be found in the list...?
                #comorbidities.append(c) # was done for creation of the list
                #print(c)
                with open(f"{run_path}/unfoundComorbidities.txt", "a") as f:
                    #write comorbities to file, seperated by comma, and as string ""
                    c = f"\"{c}\""
                    f.write(f"{c}, ")
        #return a one or multiple hot encoded vector depnding on the comorbidities in the list
        #print("len(comorbidities)", len(comorbidities))
        if len(items) == 0:
            return [0.0] * len(comorbidities)
        elif len(items) == 1:
            return [1.0 if items[0] == emb else 0.0 for emb in comorbidities]
        else:
            return [1.0 if emb in items else 0.0 for emb in comorbidities]

    def tabular_data_str_entries(items):
        one_hot_vec = tabular_data_str_one_hot(items)
        #now return a new vector with indeces of the one hot encoded vector
        return [i for i, val in enumerate(one_hot_vec) if val == 1.0]
        
    def preprocess_targets(targets):
        """
        With this function the targets are preprocessed. For now, we create one_hot encoded vectors. 0 remains 0, >0 becomes 1.
        """
        tars =  [1.0 if t > 0 else 0.0 for t in targets]
        #append one entry to the end of the list, as the last
        #this special target is only set to 1.0 if all other targets are 0.0, else 0.0
        tars.append(0.0 if sum(tars) > 0 else 1.0)
        return tars

    patient_data = []
    tabularEmbedding = torch.nn.Embedding(len(comorbidities), config.data.embedding_dim_comorbidities) #for comorbidities, not position emnbeddding added, as the order of comorbidities is not important
    for img in img_files:
        patient = label_file[label_file['image_id'] == img[:-4]]
        patient_tab_data = patient.iloc[:, 3:11].values.tolist()[0] #patient_age to nationality
        #create a copy of the tabular data, to store the plain data
        patient_tab_data_plain = patient_tab_data.copy()
        patient_targets = [float(val) if val != "bv" else np.nan for val in patient.iloc[:, 20:-1].values.tolist()[0]] #replace "bv" with nan, from "diabetic_retinopathy" to "other"
        patient_targets = preprocess_targets(patient_targets)
        #make cols 0 and 2 to floats:
        #print(patient_tab_data)
        #------------------------------------ age --------------------------------------------
        patient_tab_data[0] = float(patient_tab_data[0])
        patient_tab_data_plain[0] = str(patient_tab_data_plain[0])
        #patient_tab_data[1]: comorbidities
        #encode comorbidities depending on the number of comorbidities
        #------------------------------------ comorbidities --------------------------------------------
        if not type(patient_tab_data[1]) == str:
            patient_tab_data[1] = str(patient_tab_data[1])
            patient_tab_data_plain[1] = str(patient_tab_data_plain[1])
        comorbs = patient_tab_data[1].split(", ")
        comorbs_indices = tabular_data_str_entries(comorbs)
        #print("comorbs_indices = ",comorbs_indices)
        patient_tab_data[1] = tabularEmbedding(torch.tensor(comorbs_indices, dtype=torch.long)) #Embedding for tabular data (comorbidities)
        patient_tab_data[1] = convertTensorToList(patient_tab_data[1]) #in order to being able to concatenate to other lits (done later)
        #print(patient_tab_data[1])
        #print(len(patient_tab_data[1]))
        #------------------------------------ diabetes time --------------------------------------------
        if patient_tab_data[2] == "Não": #replace "Não" with nan
            patient_tab_data[2] = np.nan
            patient_tab_data_plain[2] = "nan"
        if patient_tab_data[2] == "1O": #wrong entry, replace "O" "0"
            patient_tab_data[2] = 10
            patient_tab_data_plain[2] = "10"
        patient_tab_data[2] = float(patient_tab_data[2]) if type(patient_tab_data[2]) != str else float(patient_tab_data[2].replace(",", ".")) #replace comma with dot
        patient_tab_data_plain[2] = str(patient_tab_data_plain[2]).replace(",", ".")
        #------------------------------------ insuline --------------------------------------------
        if patient_tab_data[3] == "yes": #one hot encoding
            patient_tab_data[3] = 1.0
        elif patient_tab_data[3] == "no":
            patient_tab_data[3] = 0.0
        else:
            patient_tab_data[3] = np.nan
            patient_tab_data_plain[3] = "nan"
        #-------------------------------- patient_sex --------------------------------------------
        if patient_tab_data[4] == 1: #one hot encoding
            patient_tab_data[4] = 0.0
            patient_tab_data_plain[4] = "M"
        elif patient_tab_data[4] == 2:
            patient_tab_data[4] = 1.0
            patient_tab_data_plain[4] = "F"
        else:
            patient_tab_data[4] = np.nan
            patient_tab_data_plain[4] = "nan"
        #-------------------------------- exam_eye --------------------------------------------
        if patient_tab_data[5] == 1: #one hot encoding
            patient_tab_data[5] = 0.0 #right eye
            patient_tab_data_plain[5] = "right eye"
        elif patient_tab_data[5] == 2:
            patient_tab_data[5] = 1.0 #left eye
            patient_tab_data_plain[5] = "left eye"
        else:
            patient_tab_data[5] = np.nan
            patient_tab_data_plain[5] = "nan"
        #-------------------------------- diabetes --------------------------------------------
        if patient_tab_data[6] == "yes": #one hot encoding
            patient_tab_data[6] = 1.0
        elif patient_tab_data[6] == "no":
            patient_tab_data[6] = 0.0
        else:
            patient_tab_data[6] = np.nan
            patient_tab_data_plain[6] = "nan"
        #remove last column (nationality) in patient_tab_data as all are "Brazil"
        patient_tab_data = patient_tab_data[:-1]
        patient_tab_data_plain = patient_tab_data_plain[:-1]
        patient_data.append({
            "img": img,
            "tab":  patient_tab_data,
            "tab_plain" : patient_tab_data_plain,
            "targets": patient_targets
        })

    np.random.seed(42)
    np.random.shuffle(patient_data)

    print("----------------------")
    print("Number of total samples: ", len(patient_data))
    patient_data = patient_data[:config.data.num_samples] if config.data.num_samples < len(patient_data) else patient_data
    print("used_samples: ", len(patient_data))
    print("----------------------")

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
        print(mode)
        return mode

    def replace_nan_entries(processed_data, reference_data):
        #replace "nan" entries in the clinical information with the mean of the column
        #print(f"Replace nan entries in {processed_data} with the mean of the column in {reference_data}")
        mean_age = np.mean([p["tab"][0] for p in reference_data if ~np.isnan(p["tab"][0])])
        #mode_comorbidities = mode([p["tab"][1] for p in reference_data if ~np.isnan(p["tab"][1])]) #TODO: delete, as solved via embedding
        mean_diabetes_time_y = np.mean([p["tab"][2] for p in reference_data if ~np.isnan(p["tab"][2])])
        mode_insuline = mode([p["tab"][3] for p in reference_data if ~np.isnan(p["tab"][3])])
        mode_patient_sex = mode([p["tab"][4] for p in reference_data if ~np.isnan(p["tab"][4])])
        mode_exam_eye = mode([p["tab"][5] for p in reference_data if ~np.isnan(p["tab"][5])])
        mode_diabetes = mode([p["tab"][6] for p in reference_data if ~np.isnan(p["tab"][6])])
        #mode_nationality = mode([p["tab"][7] for p in reference_data if p["tab"][7] != ""]) #removed for now as all are "Brazil"

        for patient in processed_data:
            for i, c_data in enumerate(patient["tab"]):
                if i == 0: #age (mean)
                    if np.isnan(c_data):
                        patient["tab"][i] = mean_age
                        patient["tab_plain"][i] = str(mean_age)
                elif i == 1: #comorbitdities (take mode)
                    continue
                elif i == 2: #diabetes time (take mean)
                    if np.isnan(c_data):
                        patient["tab"][i] = mean_diabetes_time_y
                        patient["tab_plain"][i] = str(mean_diabetes_time_y)
                elif i == 3: #insuline (take mode)
                    if np.isnan(c_data):
                        patient["tab"][i] = mode_insuline
                        patient["tab_plain"][i] = str(mode_insuline)
                elif i == 4: #patient sex (take mode)
                    if np.isnan(c_data):
                        patient["tab"][i] = mode_patient_sex
                        patient["tab_plain"][i] = str(mode_patient_sex)
                elif i == 5: #exam eye (take mode)
                    if np.isnan(c_data):
                        patient["tab"][i] = mode_exam_eye
                        patient["tab_plain"][i] = str(mode_exam_eye)
                elif i == 6: #diabetes (take mode)
                    if np.isnan(c_data):
                        patient["tab"][i] = mode_diabetes
                        patient["tab_plain"][i] = str(mode_diabetes)
                #elif i == 7: #nationality (take mode)
                    #if c_data == "":
                        #patient["tab"][i] = mode_nationality
        return

    #Split data into train and validation
    split = int(0.8 * len(patient_data))
    train_data = patient_data[:split]
    val_data = patient_data[split:]

    #write out file names of train and val data (for reload correct items in interpretability postprocessing)
    with open(f"{run_path}/train_data.txt", "w") as f:
        for item in train_data:
            f.write(f"{item['img']}\n")
    with open(f"{run_path}/val_data.txt", "w") as f:
        for item in val_data:
            f.write(f"{item['img']}\n")

    #plot gender distribution
    n_M_train = len([p["tab"][4] for p in train_data if p["tab"][4] == 0.0])
    n_F_train = len([p["tab"][4] for p in train_data if p["tab"][4] == 1.0])
    n_nan_train = len([p["tab"][4] for p in train_data if np.isnan(p["tab"][4])])
    n_M_val = len([p["tab"][4] for p in val_data if p["tab"][4] == 0.0])
    n_F_val = len([p["tab"][4] for p in val_data if p["tab"][4] == 1.0])
    n_nan_val = len([p["tab"][4] for p in val_data if np.isnan(p["tab"][4])])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].bar(["M", "F"], [n_M_train, n_F_train])
    axs[0].set_title(f"Train (n = {n_M_train + n_F_train})")
    #add note about nan entries to plot, if there are any
    if n_nan_train > 0 : axs[0].text(0, n_M_train + 10, f"nan: {n_nan_train}")
    axs[1].bar(["M", "F"], [n_M_val, n_F_val])
    axs[1].set_title(f"Validation (n = {n_M_val + n_F_val})")
    if n_nan_val > 0: axs[1].text(0, n_M_val + 10, f"nan: {n_nan_val}")
    plt.tight_layout()
    plt.savefig(f"{run_path}/gender_distribution.png")
    plt.savefig(f"{run_path}/gender_distribution.eps")

    print(train_data[0])

    train_mean, train_std = get_mean_std(train_data, cols=[0,2], key="tab")

    #2 replace the nan entries in the clinical information with the mean or mode of the training data
    replace_nan_entries(processed_data=train_data, reference_data=train_data)
    replace_nan_entries(processed_data=val_data, reference_data=train_data) #with mean or mode of data in training data!!!

    #Normalize tabular data
    if config.data.apply_znorm:
        z_norm(train_data, train_mean, train_std, cols=[0,2], key="tab")
        z_norm(val_data, train_mean, train_std, cols=[0,2], key="tab") #with mean and std of training data!!!

    #make sure, all entries in tabular data are single values, and not lists,
    #the second entry in tabular data (i.e. train_data["tab"][1]) is a list, as it is one hot encoded
    #so we now want the values to be inserted after each other
    #herein we store these values, remove the list and insert the values again
    for patient in train_data:
        tab = patient["tab"]
        tab = tab[:1] + tab[1] + tab[2:]
        patient["tab"] = tab
    for patient in val_data:
        tab = patient["tab"]
        tab = tab[:1] + tab[1] + tab[2:]
        patient["tab"] = tab

    #print len of datasets:
    print(f"Train data: {len(train_data)}")
    print(f"Validation data: {len(val_data)}")

    #write out the procesed data to .txt files
    with open(f"{run_path}/train_data_processed.txt", "w") as f:
        for item in train_data:
            f.write(f"{item}\n")
    with open(f"{run_path}/val_data_processed.txt", "w") as f:
        for item in val_data:
            f.write(f"{item}\n")

    #print one sample
    print(train_data[0])

    # v2 transforms deactivated for now
    """
    train_transforms = v2.Compose(
                [
                    v2.Resize(config.net.img_size),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(degrees=15),
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32), #FIXME: issues with this line !!!!!
                    #v2.ToDtype(torch.float32), destroys the whole image.... some bugs in here!
                    #v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
    )

    val_transforms = v2.Compose(
                [
                    v2.Resize(config.net.img_size),
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32),
                    #v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
    )

    """
    train_transforms = transforms.Compose(
            [
                transforms.Resize(config.net.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #typical values computed from imagenet
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

    #plot dataset analysis
    #get all targets in one vector
    train_targets = [item["targets"] for item in train_data] #instead get_targets(train_dataset)
    val_targets = [item["targets"] for item in val_data]

    #sum up all targets
    train_targets = np.sum(train_targets, axis=0)
    val_targets = np.sum(val_targets, axis=0)

    create_bar_graph(train_targets, "Train", relative=True, filename=f"{run_path}/train_targets_bar_plot")
    create_bar_graph(val_targets, "Val", relative=True, filename=f"{run_path}/val_targets_bar_plot")

    #plot and save some examples
    fig, axs = plt.subplots(4, 3, figsize=(40, 30))
    for i in range(12):
        sample = train_dataset[i]
        img = sample["image"]
        tab = sample["tabular"]
        targets = sample["targets"]
        img_name = sample["img_name"]
        ax = axs[i // 3, i % 3]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"{img_name}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{run_path}/example_images.png")


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
        elif name == "SwinViTMLP":
            return SwinViTMLPNet(
                in_channels=config.net.in_channels,
                img_size=config.net.img_size,
                patch_size=config.net.patch_size,
                spatial_dims=config.net.spatial_dims,
                num_classes=config.net.num_classes,
                num_clinical_features=config.net.num_clinical_features+config.data.embedding_dim_comorbidities-1, #solved
                embed_dim=config.net.embed_dim,
                num_heads=config.net.num_heads_in_layer,
                depths=config.net.depths,
                window_size=config.net.window_size,
                dropout_rate=config.net.dropout_rate,
                qkv_bias=config.net.qkv_bias,
                only_vision=config.net.only_vision,
                only_clinical=config.net.only_clinical,
            )
        elif name == "SwinViTMLP_torch":
            return SwinViTMLPNet_torch(
                in_channels=config.net.in_channels,
                img_size=config.net.img_size,
                patch_size=config.net.patch_size,
                spatial_dims=config.net.spatial_dims,
                num_classes=config.net.num_classes,
                num_clinical_features=config.net.num_clinical_features+config.data.embedding_dim_comorbidities-1, #solved
                embed_dim=config.net.embed_dim,
                num_heads=config.net.num_heads_in_layer,
                depths=config.net.depths,
                window_size=config.net.window_size,
                dropout_rate=config.net.dropout_rate,
                only_vision=config.net.only_vision,
                only_clinical=config.net.only_clinical,
            )
        else:
            raise ValueError(f"Architecture {name} not found")


    model = Net(config.net.name)
    print(model)
    #save the model architecture
    with open(os.path.join(run_path, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))
        
    # Define the optimizer adamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)

    # Define the loss function binary cross entropy
    loss_function = torch.nn.BCELoss().to(config.device)

    #training parameters
    epochs = config.train.epochs
    val_interval = config.val.interval
    model.to(config.device)

    def precision(y_true, y_pred):
        """
        Compute the precision
        """
        TP = sum([1 if y_true[i] == 1 and y_pred[i] == 1 else 0 for i in range(len(y_true))])
        FP = sum([1 if y_true[i] == 0 and y_pred[i] == 1 else 0 for i in range(len(y_true))])
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    def recall(y_true, y_pred):
        """
        Compute the recall
        """
        TP = sum([1 if y_true[i] == 1 and y_pred[i] == 1 else 0 for i in range(len(y_true))])
        FN = sum([1 if y_true[i] == 1 and y_pred[i] == 0 else 0 for i in range(len(y_true))])
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    def compute_f1_score(gt, pred, num_classes=13):
        """
        f1 score for each class
        we take 0.5 as threshold for now.
        this must be changed for a "correct" f1 score
        """
        with torch.no_grad():
            f1_scores = np.array([])
            gt_np = gt
            #ma to 0 1
            pred_np = pred
            for i in range(num_classes):
                #use sklearn learn to compute the f1 score
                if sum(gt_np[:, i].tolist()) == 0:
                    f1_scores = np.append(f1_scores, np.nan)
                pred_np_cls = [1 if p > 0.5 else 0 for p in pred_np[:, i].tolist()]
                f1_scores = np.append(f1_scores, f1_score(gt_np[:, i].tolist(), pred_np_cls))
        return f1_scores

    def compute_AUCs(gt, pred, num_classes=13):
        with torch.no_grad():
            AUROCs = []
            gt_np = gt
            pred_np = pred
            for i in range(num_classes):
                #handle if all values are 0 in gt_np[:, i].tolist() "error was: "Only one class present in y_true. ROC AUC score is not defined in that case.
                #this can happen, if the class is not present in the validation set batch
                #this cannot happen, when we compute the auc for one row as then always at least one class will be present
                #but as we want classwise auc, we need to handle this case here
                if sum(gt_np[:, i].tolist()) == 0:
                    AUROCs.append(np.nan)
                    print(f"AUROC for class i = {i} is set to np.nan")
                else:
                    AUROCs.append(roc_auc_score(gt_np[:, i].tolist(), pred_np[:, i].tolist()))
            #print("AUROCs", AUROCs)
        return AUROCs

    def compute_ACCs(gt, pred, num_classes=13): #no good metric for our case
        with torch.no_grad():
            ACCs = []
            gt_np = gt
            pred_np = pred
            for i in range(num_classes):
                pred_np_cls = [1 if p > 0.5 else 0 for p in pred_np[:, i].tolist()]
                ACCs.append(np.mean(np.array(gt_np[:, i].tolist()) == np.array(pred_np_cls)))
        return ACCs

    def train(train_loader, optimizer, loss_func):
        model.train()
        running_loss = 0.0
        print_step_interval = max(len(train_loader) // 10, 1)
        for steps, data in enumerate(train_loader):
            if steps % print_step_interval == 0:
                print(f"Step {steps}/{len(train_loader)}")
            img = data["image"].to(config.device)
            tab = data["tabular"].to(config.device)
            targets = data["targets"].to(config.device)
            optimizer.zero_grad()
            output = model(tab, img)
            #print("output", output.flatten())
            #print("targets", targets.flatten())
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(val_loader, loss_func, save_model=True):
        model.eval()
        running_loss = 0.0
        targets_in = torch.tensor([], dtype=torch.float32, device=config.device)
        preds_cls = torch.tensor([], dtype=torch.float32, device=config.device)
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                img = data["image"].to(config.device)
                tab = data["tabular"].to(config.device)
                targets = data["targets"].to(config.device)
                output_probs= model(tab, img)
                loss = loss_func(output_probs, targets)
                running_loss += loss.item()
                targets_in = torch.cat([targets_in, targets], dim=0)
                preds_cls = torch.cat([preds_cls, output_probs], dim=0)
            auc = compute_AUCs(targets_in, preds_cls, config.net.num_classes)
            f1_scores = compute_f1_score(targets_in, preds_cls, config.net.num_classes)
            #compute mean values, expect nana values
            mean_auc = np.nanmean(auc)
            #print("f1_scores", f1_scores)
            mean_f1_scores = np.nanmean(f1_scores)
        if save_model:
            #store model together with epoch number and optimizer
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(run_path, f"model_last_eval.pth"))
        return running_loss / len(val_loader), mean_auc, auc, mean_f1_scores, f1_scores

    # Training loop
    best_auc = 0
    best_f1 = 0
    best_loss = np.inf
    for epoch in range(epochs):
        train_loss = train(train_loader, optimizer, loss_function)
        writer.add_scalar('Loss/train', train_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss}")
        if (epoch + 1) % val_interval == 0:
            val_loss, mean_auc, auc, mean_f1_scores, f1_scores = validate(val_loader, loss_function)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('AUC/val', mean_auc, epoch)
            writer.add_scalar('F1/val', mean_f1_scores, epoch)
            print(f"Epoch {epoch+1}/{epochs}: Validation Loss: {val_loss}", f"Mean AUC: {mean_auc}, Mean f1: {mean_f1_scores}")
            #store best auc model
            if mean_auc >= best_auc:
                best_auc = mean_auc
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(run_path, f"model_best_auc.pth"))
            #store best f1 model
            if mean_f1_scores >= best_f1:
                best_f1 = mean_f1_scores
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(run_path, f"model_best_f1.pth"))
            #store lowest loss model
            if val_loss <= best_loss:
                best_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(run_path, f"model_best_loss.pth"))
        
    #save the final model
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(run_path, f"model_final.pth"))

    #Further evaluation
    def performance(loader):
        """
        AUC and ACC details.
        """
        print("Evaluation:")
        val_loss, mean_auc, auc, mean_f1_scores, f1_scores = validate(loader, loss_function, save_model=False)

        Evaluation_statistics = "Mean AUC : {}, Mean Loss : {}\n\nMean test AUC for each class in 14 disease categories\
            :\n\ndiabetic_retinopathy: {}\nmacular_edema: {}\nscar: {}\nnevus: \
            {}\namd: {}\nvascular_occlusion: {}\nhypertensive_retinopathy: {}\ndrusens: \
            {}\nhemorrhage: {}\nretinal_detachment: {}\nmyopic_fundus: {}\nincreased_cup_disc: \
            {}\nother: {}\nNo_Finding: {}".format(
                mean_auc, val_loss,
                auc[0],
                auc[1],
                auc[2],
                auc[3],
                auc[4],
                auc[5],
                auc[6],
                auc[7],
                auc[8],
                auc[9],
                auc[10],
                auc[11],
                auc[12],
                auc[13],
            )
        Evaluation_statistics += "\n\nMean f1 : {}, Mean Loss : {}\n\nMean test f1 score for each class in 14 disease categories\
            :\n\ndiabetic_retinopathy: {}\nmacular_edema: {}\nscar: {}\nnevus: \
            {}\namd: {}\nvascular_occlusion: {}\nhypertensive_retinopathy: {}\ndrusens: \
            {}\nhemorrhage: {}\nretinal_detachment: {}\nmyopic_fundus: {}\nincreased_cup_disc: \
            {}\nother: {}\nNo_Finding: {}".format(
                mean_f1_scores, val_loss,
                f1_scores[0],
                f1_scores[1],
                f1_scores[2],
                f1_scores[3],
                f1_scores[4],
                f1_scores[5],
                f1_scores[6],
                f1_scores[7],
                f1_scores[8],
                f1_scores[9],
                f1_scores[10],
                f1_scores[11],
                f1_scores[12],
                f1_scores[13],
            )
        print(Evaluation_statistics)
        return Evaluation_statistics

    #PERFORMANCE
    #load best auc model for performance evaluation
    checkpoint = torch.load(os.path.join(run_path, f"model_best_auc.pth"))
    model.load_state_dict(checkpoint["state_dict"])

    val_stats = performance(val_loader)
    train_stats = performance(train_loader)

    #write out infos to file "performance.txt"
    with open(f"{run_path}/performance.txt", "w") as f:
        f.write(f"Validation statistics:\n{val_stats}\n\nTrain statistics:\n{train_stats}\n\n")

    #Confusion Matrix, skipped for now
    def ConfusionMatrix(loader, filenamepostfix: str="val"):
        """ 
        Classification report for a given net and dataset.
        loader = validation or train loader
        """
        model.eval()

        # Evaluate best model
        y_gt = torch.tensor([], dtype=torch.float32, device=config.device)
        y_pred = torch.tensor([], dtype=torch.float32, device=config.device)
        with torch.no_grad():
            for i, data in enumerate(loader):
                img = data["image"].to(config.device)
                tab = data["tabular"].to(config.device)
                targets = data["targets"].to(config.device)
                outputs = model(tab, img)
                #set the threshold to 0.5 and change outputs directily
                outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=config.device), torch.tensor(0.0, device=config.device))
                #As multiple classes can be predicted, we need to find the indeces of the classes with the highest probability
                list_indices_gt = [[i for i in range(0,len(vec)) if vec[i] == torch.max(vec)] for vec in targets]
                list_indices_pred = [[i for i in range(0,len(vec)) if vec[i] == torch.max(vec)] for vec in outputs]
                for entry in list_indices_gt:
                    y_gt = torch.cat([y_gt, torch.tensor(entry, dtype=torch.float32, device=config.device)], dim=0)
                for entry in list_indices_pred:
                    y_pred = torch.cat([y_pred, torch.tensor(entry, dtype=torch.float32, device=config.device)], dim=0)

        #detach
        y_gt = y_gt.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
            
        print("Classification Report")
        print("--------------------")
        print("y_gt", y_gt)
        print("len(y_gt)", len(y_gt))
        print("y_pred", y_pred)
        print("len(y_pred)", len(y_pred))

        print(classification_report(
            y_gt,
            y_pred,
            target_names=[d.name for d in Diseases])
        )

        cm = confusion_matrix(
            y_gt,
            y_pred,
            normalize='true',
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[d.name for d in Diseases],
        )
        disp.plot(ax=plt.subplots(1,1,facecolor='white')[1])
        plt.xticks(rotation=45, ha='right')
        if os.path.isdir(f"{run_path}/performance") == False:
            os.mkdir(f"{run_path}/performance")
        plt.savefig(f"{run_path}/performance/confusion_{filenamepostfix}.png")
        plt.savefig(f"{run_path}/performance/confusion_{filenamepostfix}.eps")
        return

    #more to be done in brset_interpretability_postprocessing.py and performance.py

if __name__ == "__main__":
    main()

