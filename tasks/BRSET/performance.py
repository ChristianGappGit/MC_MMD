"""
A Brazilian Multilabel Ophthalmological Dataset (BRSET)
downloaded data from https://physionet.org/content/brazilian-ophthalmological/1.0.0/
github: https://github.com/luisnakayama/BRSET/tree/main

Data:
    2D images
    tabular data
Task:
    Multilabel Classification

Input: (from output_path)
    - images, tabular data 
    - labels

Output:
    - performance metrics

Usage: python3 perforamnce.py output/run_x/ model_name
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")

import torch
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
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

from brset_multimodal_classification import MultimodalDataset, Diseases


def main(run_path, model_name):
    run_path = run_path[:-1] if run_path[-1] == "/" else run_path
    output_dir = f"{run_path}/performance/{model_name.split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)

    #----------- Load Configuration File -----------#
    config_file = f"{run_path}/default.yaml"
    config = load_config(config_file)
    torch.backends.cudnn.benchmark = True
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
    with open(f"{run_path}/val_data_processed.txt", "r") as f:
        val_data = f.readlines()
    val_data = [eval(x) for x in val_data]

    val_transforms = transforms.Compose(
                [
                transforms.Resize(config.net.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
    )

    val_params = {
        "batch_size": config.val.batch_size,
        "shuffle": False, #config.val.shuffle, overwritten as we never want to shuffle the validation set
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
    #load best auc model for performance evaluation
    checkpoint = torch.load(os.path.join(run_path, model_name))
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Model loaded from {run_path}/{model_name} with epoch {checkpoint['epoch']}")
    model.to(config.device)
    model.eval()

    # Define the loss function binary cross entropy
    loss_function = torch.nn.BCELoss().to(config.device)

    def compute_f1_score(gt, pred, num_classes=13):
        """
        f1 score for each class
        we take 0.5 as threshold for now.
        this must be changed for a "correct" f1 score
        """
        with torch.no_grad():
            f1_scores = np.array([])
            gt_np = gt
            pred_np = pred
            #device cpu!
            gt_np = gt_np.cpu().numpy()
            pred_np = pred_np.cpu().numpy()
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
            #device cpu!
            gt_np = gt_np.cpu().numpy()
            pred_np = pred_np.cpu().numpy()
            for i in range(num_classes):
                #print("gt_np[:,i].tolist()", gt_np[:, i].tolist())
                #print("pred_np[:,1].tolist()",pred_np[:, i].tolist())
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

    from tqdm import tqdm
    # Validation
    def evaluate(loader, loss_func):
        model.eval()
        running_loss = 0.0
        targets_in = torch.tensor([], dtype=torch.float32, device=config.device)
        preds_cls = torch.tensor([], dtype=torch.float32, device=config.device)
        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
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
        return running_loss / len(loader), mean_auc, auc, mean_f1_scores, f1_scores

    #----------- Performance -----------#
    def performance(loader):
            """
            AUC and ACC details.
            """
            print("Evaluation:")
            val_loss, mean_auc, auc, mean_f1_scores, f1_scores = evaluate(loader, loss_function)

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

    val_stats = performance(val_loader)

    #Write performance to file
    with open(f"{output_dir}/performance.txt", "w") as f:
        f.write("Performance on Validation Set\n")
        f.write(val_stats)
    print("Performance written to", f"{output_dir}/performance.txt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 performance.py <output/run_x/> <model_name>")
        sys.exit(1)
    output_path = sys.argv[1]
    model_name = sys.argv[2]
    main(output_path, model_name)