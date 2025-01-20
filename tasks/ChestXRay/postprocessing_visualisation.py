"""
File to visualise the results of the interpretability_postprocessing.py script.
Run this script with "python postprocessing_visualisation.py <dir>" in the terminal, where 
<dir> is the directory of the results you want to process.
For instance: "python postprocessing_visualisation.py run_39/interpretability
ls <dir>" should contain the following:
"img_name0"
"img_name1"
...
"img_nameN-1"
"GradCAM_text_cbar.html"
"OccSens_text_cbar.html"
       ls <dir>/img_name0 should contain the following:
         "img_name0_GradCAM_text.html"
         "img_name0_OccSens_text.htm"
         "img_name0_interpretability_vision.png" (*.pdf)

The *.html files together with the *.png files in one directory are summarised in a html file.
and saved as "interpretability.html" in the same directory.

The path to the excel file containing the targets of the items are hard coded in this script.
Change the path in the function "get_targets" if necessary.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from enum import Enum


def main(perform_dir: str):

    class Diseases(Enum): 
        """
        Enum class for the diseases in the dataset
        """
        Atelectasis =                   [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Cardiomegaly =                  [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Consolidation =                 [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Edema =                         [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Enlarged_Cardiomediastinum =    [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Fracture =                      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Lung_Lesion =                   [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Lung_Opacity =                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
        No_Finding =                    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
        Pleural_Effusion =              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
        Pleural_Other =                 [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
        Pneumonia =                     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
        Pneumothorax =                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
        Support_Devices =               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]

    def get_targets_pred(img_name):
        #get targets_pred from interpretability_information.txt file located in perform_dir
        #ensure there is no file ending
        img_name = img_name.split(".")[0]
        """images are stored as following:
        CXR127_IM-0181-1001
        GT:    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        Pred:  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        Probs: ['0.0005', '0.0003', '0.0002', '0.0002', '0.0007', '0.0009', '0.0005', '0.0034', '0.9966', '0.0014', '0.0012', '0.0003', '0.0019', '0.0010']

        CXR1288_IM-0189-1001
        GT:    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        Pred:  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        Probs: ['0.5483', '0.0009', '0.0143', '0.0029', '0.0031', '0.0032', '0.0313', '0.9966', '0.0076', '0.0231', '0.0054', '0.0093', '0.0041', '0.0118']
        """
        with open(f"{perform_dir}/interpretability_information.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(img_name):
                    #find next line that starts with Pred
                    index = lines.index(line)
                    while not lines[index].startswith("Pred"):
                        index += 1
                    #get targets_pred
                    targets_pred = lines[index].split("[")[1].split("]")[0].split(", ")
                    targets_pred = [int(item) for item in targets_pred]
                    break
        return targets_pred

    def get_importance_ratios_str(img_name):
        """
        get the importance ratios from the interpretability_information.txt file
        stored as following:
        Importance per class (text): [xi0, xi1,...xin]
        Importance per class (vision): [xv0, xv1,...xvn]
        Importance mean ratio (text : vision) = weight_text : weight_vision
        """
        with open(f"{perform_dir}/interpretability_information.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(img_name):
                    #find next line that starts with Importance ratio
                    index = lines.index(line)
                    while not lines[index].startswith("Importance per class (text)"):
                        index += 1
                    #get whole line as string
                    importance_text = lines[index].split(": ")[1].split("\n")[0].strip()
                    index += 1
                    importance_vision_occ = lines[index].split(": ")[1].split("\n")[0].strip()
                    index += 1
                    importance_ratio_occ = lines[index].split("= ")[1].split("\n")[0].strip()
                    index += 1
                    importance_vision_cam = lines[index].split(": ")[1].split("\n")[0].strip()
                    index += 1
                    importance_ratio_cam = lines[index].split("= ")[1].split("\n")[0].strip()
                    break
        return importance_text, importance_vision_occ, importance_ratio_occ, importance_vision_cam, importance_ratio_cam

    def process_item(img_name, dir, dataset): 
        """
        process the directory <dir> of the item <img_name>
        and store all data to single hmt file <img_name_postprocessing.html>
        """
        print("Processing", img_name, "in", dir)
        #get all files in dir
        files = os.listdir(dir)
        files = [file for file in files if file.endswith(".html") or file.endswith(".png")]
        files.sort()
        #print(files)
        #get GradCAM file
        gradcam_file = [file for file in files if file.endswith("GradCAM_text.html")][0]
        #get OccSens file
        occsens_file = [file for file in files if file.endswith("OccSens_text.html")][0]
        #get vision file
        vision_file = [file for file in files if file.endswith("interpretability_vision.png")][0]
        #get colorbar files from ..dir above
        #print(os.listdir(perform_dir))
        colorbar_gradcam_file = [file for file in os.listdir(perform_dir) if file.endswith("GradCAM_text_cbar.html")][0]
        colorbar_occsens_file = [file for file in os.listdir(perform_dir) if file.endswith("OccSens_text_cbar.html")][0]

        #get targets of img_name
        #print("img_name", img_name)
        targets = dataset.get_targets(img_name+".png")
        targets_pred = get_targets_pred(img_name)

        #get importance ratios
        importance_text, importance_vision_occ, importance_ratio_occ, importance_vision_cam, importance_ratio_cam = get_importance_ratios_str(img_name)

        #get the Diseases from the targets
        Diseases_list = [disease for disease in Diseases]
        diseases = [Diseases_list[i].name for i in range(len(targets)) if targets[i] == 1]
        diseases_pred = [Diseases_list[i].name for i in range(len(targets_pred)) if targets_pred[i] == 1]

        #store png and html files in one html file
        html_file = f"{dir}/{img_name}_postprocessing.html"
        with open(html_file, "w") as f:
            f.write(f"<h1>{img_name}</h1>")
            f.write(f"<h2>Importance</h2>")
            f.write(f"<h3>Importance text:                 {importance_text}</h3>")
            f.write(f"<h3>Importance vision:               {importance_vision_occ} (OCC sens CG)</h3>")
            f.write(f"<h3>Importance vision:               {importance_vision_cam} (CAM)</h3>")
            f.write(f"<h3>Importance ratio (text:vision) = {importance_ratio_occ} (OCC sens CG)</h3>")
            f.write(f"<h3>Importance ratio (text:vision) = {importance_ratio_cam} (CAM)</h3>")
            f.write(f"<h2>Interpretability Vision</h2>")
            f.write(f"<h4>(red parts imply importance)</h4>")
            f.write(f"<img src='{vision_file}' width='1230' height='205'>")
            f.write("<br>")
            f.write(f"Occ_sens (MONAI): specific for first class with label 1 (pred).</br>")
            f.write(f"Occ_sens (CG): mean of all changes with respect to all classes.")
            f.write(f"<h2>Interpretability Text</h2>")
            f.write(f"<h4>(red parts imply importance)</h4>")
            f.write(f"<h3>GradCAM</h3>")
            #add html file as html code
            with open(f"{dir}/{gradcam_file}", "r") as g:
                f.write(g.read())
            f.write("<br>")
            with open(f"{perform_dir}/{colorbar_gradcam_file}", "r") as g:
                f.write("<br>")
                f.write(g.read())
            f.write(f"<h3>OccSens</h3>")
            #add html file as html code
            with open(f"{dir}/{occsens_file}", "r") as g:
                f.write(g.read())
            f.write("<br>")
            with open(f"{perform_dir}/{colorbar_occsens_file}", "r") as g:
                f.write("<br>")
                f.write(g.read())
            f.write(f"<h2>Targets </h2>")
            f.write(f"<h3>GT:   {targets}</h3>")
            f.write(f"<h3>Pred: {targets_pred}</h3>")
            f.write(f"<h2>Diseases</h2>")
            f.write(f"<h3>GT:   {diseases}</h3>")
            f.write(f"<h3>Pred: {diseases_pred}</h3>")

    class Dataset():
        def __init__(self, dataframe):
            self.data = dataframe
            self.report_summary = self.data.report
            self.img_name = self.data.id
            self.targets = self.data.list

        def get_targets(self, img_name):
            #return targets of img_name, where self.img_name contains the img_name
            #print("self.img_name", self.img_name)
            #print("img_name", img_name)
            index = np.where(self.img_name == img_name)[0]
            #print(index)
            return np.array(self.targets[index])[0]
        
        def __len__(self):
            return len(self.report_summary)
        
        def __getitem__(self, index):
            name = self.img_name[index].split(".")[0]
            targets = np.array(self.targets[index])
            return {
                "name": name,
                "targets": targets
            }


    def load_txt_gt(path):
        txt_gt = pd.read_csv(path)
        txt_gt["list"] = txt_gt[txt_gt.columns[2:]].values.tolist()
        txt_gt = txt_gt[["id", "report", "list"]].copy()
        return txt_gt
    
    def importance(disease):
        """
        get the importance of the disease
        assumed to be stored in the interpretability_information.txt file as following:
        EXAMPLE:
        Mean importance per class (text):            [0.81, 0.79, 0.75, 0.62, 0.79, 0.75, 0.85, 0.86, 0.78, 0.72, 0.61, 0.84, 0.56, 0.71]
        Mean importance per class (vision):          [0.19, 0.21, 0.25, 0.38, 0.21, 0.25, 0.15, 0.14, 0.22, 0.28, 0.39, 0.16, 0.44, 0.29]
        Mean importance ratio (text : vision) = 0.74 : 0.26
        """
        if disease == "all":
            with open(f"{perform_dir}/interpretability_information.txt", "r") as f:
                lines = f.readlines()
                importance_ratio = None
                for line in lines:
                    if line.startswith("Mean importance ratio (text : vision)"):
                        importance_ratio = lines[lines.index(line)].split("= ")[1].split("\n")[0].strip()
                        #extract text and vision importance from ratio
                        importance_text = float(importance_ratio.split(" : ")[0])
                        importance_vision = float(importance_ratio.split(" : ")[1])
                        break
                if importance_ratio is None:
                    print("Error: Importance ratio not found in interpretability_information.txt. Exiting...")
                return importance_text, importance_vision
        else: 
            index = Diseases[disease].value.index(1)
            with open(f"{perform_dir}/interpretability_information.txt", "r") as f:
                lines = f.readlines()
                importance_text = None
                importance_vision = None
                for line in lines:
                    if line.startswith("Mean importance per class (text)"):
                        importance_text = lines[lines.index(line)].split(": ")[1].split("\n")[0].strip()
                        #remove brackets, if present
                        if importance_text.startswith("["):
                            importance_text = importance_text[1:]
                        if importance_text.endswith("]"):
                            importance_text = importance_text[:-1]
                        importance_text = float(importance_text.split(", ")[index])
                    if line.startswith("Mean importance per class (vision)"):
                        importance_vision = lines[lines.index(line)].split(": ")[1].split("\n")[0].strip()
                        #remove brackets, if present
                        if importance_vision.startswith("["):
                            importance_vision = importance_vision[1:]
                        if importance_vision.endswith("]"):
                            importance_vision = importance_vision[:-1]
                        importance_vision = float(importance_vision.split(", ")[index])
                    if importance_text is not None and importance_vision is not None: #both found
                        break
                if importance_text is None or importance_vision is None:
                    print("Error: Importance text or vision not found in interpretability_information.txt. Exiting...")
                return importance_text, importance_vision

    #get last word from directory and chevk if it is test, train or val
    if perform_dir.split("/")[-1].endswith("test"):
        path_data_dir = '/home/christian/data/TransCheX/monai_data/monai_data/dataset_proc/test.csv'
    elif perform_dir.split("/")[-1].endswith("train"):
        path_data_dir = '/home/christian/data/TransCheX/monai_data/monai_data/dataset_proc/train.csv'
    elif perform_dir.split("/")[-1].endswith("val"):
        path_data_dir = '/home/christian/data/TransCheX/monai_data/monai_data/dataset_proc/validation.csv'
    else:
        print("Error: Directory does not end with test, train or val. Make sure to have the correct directory. Exiting...")
        return
    current_txt_gt = load_txt_gt(path_data_dir)
    current_dataset = Dataset(current_txt_gt)

    #get all names of folders in perform_dir (without files)
    img_names = os.listdir(perform_dir)
    #remove files from img_names
    img_names = [item for item in img_names if not item.endswith(".html") and item.startswith("CXR")]

    for item in img_names:
        process_item(item, f"{perform_dir}/{item}", current_dataset)

    #create a html page with all Disease names with links to all html files in the class
    diseases = [disease.name for disease in Diseases]
    html_file = f"{perform_dir}/interpretability.html"
    with open(html_file, "w") as f:
        f.write("<h1>Interpretability</h1>")
        #explain colors green, blue and red, highligh in color
        f.write("<h2>Color Explanation</h2>")
        f.write("<ul>")
        f.write("<li><a style='color:green'>Green:</a> Correctly classified.</li>")
        f.write("<li><a style='color:blue'>Blue:</a> Some classes detected correctly, some either wrong or not.</li>")
        f.write("<li><a style='color:red'>Red:</a> Completely wrong classified.</li>")
        f.write("<h2>Disease (Importance Text : Importance Vision)</h2>")
        text_importance, vision_importance = importance("all")
        f.write(f"<h2>MEAN importance = {text_importance} : {vision_importance} </h2>")
        for disease in diseases:
            text_importance_class, vision_importance_class = importance(disease)
            f.write(f"<h2>{disease} ({text_importance_class} : {vision_importance_class})</h2>")
            f.write("<ul>")
            for item in img_names:
                targets = current_dataset.get_targets(item+".png")
                #only add item to list if disease is in targets
                if targets[Diseases[disease].value.index(1)] == 1:
                    targets_pred = get_targets_pred(item)
                    #highlight green if targets and targets_pred are equal, highlight blue, if at least one index is the same, else red
                    if targets == targets_pred:
                        color = "green"
                    elif sum(np.array(targets) * np.array(targets_pred)) > 0:
                        color = "blue"
                    else:
                        color = "red"
                    f.write(f"<li><a href='{item}/{item}_postprocessing.html' style='color:{color}'>{item}</a></li>")
            f.write("</ul>")

if __name__ == "__main__":
    main(sys.argv[1])