#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #setting environmental variable "CUDA_DEVICE_ORDER"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #TODO: change, if multiple GPU needed
os.system("echo Selected GPU: $CUDA_VISIBLE_DEVICES")
import torch
import numpy as np
import pandas as pd
from enum import Enum
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset #not needed... but possible
from monai.config import print_config
from monai.utils import set_determinism
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)
#from pytorch_grad_cam import GradCAM as pytorchGradCAM #own version used (see interpretability/cam.py)
from monai.visualize import (
    GradCAM as monaiGradCAM,
#    OcclusionSensitivity as monaiOcclusionSensitivity,
    GuidedBackpropGrad
)
from peft import get_peft_model, LoraConfig
from monai.networks.utils import eval_mode
from transformers import BertTokenizer
import loralib as lora
from ViT_TexT import ImageTextNet
from ResNetText import ResNetLLaMAII
from config import Config, load_config

from llama import Tokenizer as LLaMATokenizer 

import sys
print("sys.argv", sys.argv)
print(len(sys.argv))
if len(sys.argv) != 2:
    print("Usage: python performance.py <performance_dir>")
    sys.exit(1)

# own imports for interpretability
from interpretability_ChestXRay.occlusion_sensitivity import OcclusionSensitivity, OcclusionSensitivityText, OcclusionSensitivityImage
from interpretability_ChestXRay.cam import GradCAM, GradCAMText, GradCAMCross

def main(perform_dir: str):

    #----------- Download and pre-process dataset -----------#

    #DONE
    datadir = "./monai_data"
    #if not os.path.exists(datadir):
    #    raise ValueError("Please download the dataset from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data and place it in the monai_data folder")


    #----------- Print Configurations  -----------#
    torch.backends.cudnn.benchmark = True

    print_config()

    #perform_dir = sys.argv[1] #argutment passed to main() function
    model_dir = f"{perform_dir}/best_model" #must exist
    model_name = "transchex.pt" #must exist
    lora_model_dir = f"{perform_dir}/lora_model" #must exist
    lora_model_name = "transchex_lora.pt" #must exist
    output_dir = f"{perform_dir}/interpretability" #can exist, otherwise created
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    #----------- Load Configuration File -----------#
    config_file = f"{perform_dir}/default_llama.yaml"
    config = load_config(config_file)
    print(config)

    #----------- Setup dataloaders and transforms for training/validation/testing -----------#
    class MultiModalDataset(Dataset): #Cachedataset possible, but not urgently needed for 2D images
        def __init__(self, dataframe, tokenizer, tokenizer_type, parent_dir, max_seq_length=config.text_max_seq_length): #FIXME: max_seq_length set to proper value
            self.max_seq_length = max_seq_length
            self.tokenizer = tokenizer
            self.tokenizer_type = tokenizer_type
            self.data = dataframe
            self.report_summary = self.data.report
            self.img_name = self.data.id
            self.targets = self.data.list

            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            self.parent_dir = parent_dir

        def __len__(self):
            return len(self.report_summary)

        def encode_features_bert(self, sent, max_seq_length):
            tokens = self.tokenizer.tokenize(sent.strip()) #bert tokenizer
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[: (max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            #print("tokens", tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens) #bert
            #input_ids = convert_tokens_to_ids(tokens) #llama, no function implemented yet.... as not needed?!
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                segment_ids.append(0)
                tokens.append("**NULL**")
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            return input_ids, segment_ids, tokens

        def encode_features_llama(self, sent, max_seq_length): #different from function in Multimoldal_ImageText_ChestXRay_Lung_Class..py
            #llama tokenization
            tokens = []
            return_tokens = []
            #split sent to vec of words
            sent = sent.split()
            #print(sent)
            #print("encode(**NULL**)", self.tokenizer.encode("**NULL**", bos=False, eos=False))
            for x in sent:
                #print("x",x)
                nums = self.tokenizer.encode(x, bos=False, eos=False)
                #print("nums", nums)
                #print("")
                tokens.append(nums)
                return_toks = []
                for i, t in enumerate(nums):
                    tok = self.tokenizer.decode(t)
                    if i != 0:
                        tok = "##" + tok
                    return_toks.append(tok)
                #print("return_toks", return_toks)
                #print("\n\n")
                return_tokens.append(return_toks)
            #print("tokens", tokens)
            #print("")
            #print("return_tokens", return_tokens)
            #covert to single vector
            tokens = [item for sublist in tokens for item in sublist]
            #print("tokens", tokens)
            return_tokens = [item for sublist in return_tokens for item in sublist]

            assert(len(tokens) == len(return_tokens))  #old, not true anymore

            #now tokens are encoded properly
            if len(tokens) > max_seq_length: #-2 when adding cls and sep tokens
                tokens = tokens[: (max_seq_length)] #-2
                return_tokens = return_tokens[: (max_seq_length)] #-2
            #tokens = [0] + tokens + [2] #TODO: check if correct (and add: tokens = ["[CLS]"] + tokens + ["[SEP]"]) ??
            #return_tokens = ["[CLS]"] + tokens + ["[SEP]"] #llama not pretrained with cls and sep tokens.
            input_ids = tokens

            #print("input_ids", input_ids)
            #print("type_input_ids", type(input_ids))
            #print("type_input_ids[0]", type(input_ids[0]))  #correct

            segment_ids = [0] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                segment_ids.append(0)
                return_tokens.append("**NULL**")    #TODO: CHECK if correct "**NULL**"
            assert len(return_tokens) == max_seq_length  #FIXME: could be > ... what to do then (increase max_seq_length?)
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
            #print("return_tokens", return_tokens)
            return input_ids, segment_ids, return_tokens

        def __getitem__(self, index):
            name = self.img_name[index].split(".")[0]
            img_address = os.path.join(self.parent_dir, self.img_name[index])
            image = Image.open(img_address)
            images = self.preprocess(image)
            report = str(self.report_summary[index])
            report = " ".join(report.split())
            if self.tokenizer_type == "bert":
                input_ids, segment_ids, tokens = self.encode_features_bert(
                    report, self.max_seq_length,
                )
            else:
                input_ids, segment_ids, tokens = self.encode_features_llama(
                    report, self.max_seq_length,
                )
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            targets = torch.tensor(self.targets[index], dtype=torch.half) #TODO: changed to half precision
            #print("tokens", tokens)
            return {
                "ids": input_ids,
                "segment_ids": segment_ids,
                "name": name,
                "targets": targets,
                "images": images,
                "tokens": tokens,
            }

    #-----------  Setup the model directory, tokenizer and dataloaders -----------#
    def preprocess_text(text, class_names):
        """
        preprocess text data:
        delete all words that are equal to class names
        """
        #print("class_names", class_names)
        #print("text[0]", text[0])
        class_names = [name.lower() for name in class_names]
        #add class names with "-" and "_" to class_names (seperate words)
        class_names_add = []
        for name in class_names:
            if name.find("-") > 0:
                name_split = name.split("-")
                for name_split_item in name_split:
                    if name_split_item != "no": class_names_add.append(name_split_item)
            if name.find("_") > 0:
                name_split = name.split("_")
                for name_split_item in name_split:
                    if name_split_item != "no": class_names_add.append(name_split_item)
        class_names += class_names_add
        #print("class_names", class_names)
        text = [text_line.lower().split() for text_line in text]
        #print("text[0]", text[0])
        text_processed = []
        for textline in text:
            #print("textline", textline)
            text_line_processed = [] #reset for each textline
            for word in textline:
                #print(word)
                #remove "." and "," from word:
                word_untouched = word
                post_fix = ""
                if word.find(".") > 0:
                    word = word.replace(".", "")
                    post_fix = "."
                if word.find(",") > 0:
                    word = word.replace(",", "")
                    post_fix = "," #note that both "." and "," will never be in the same word
                #print(word)
                #print(word_untouched)
                if word not in class_names:
                    text_line_processed.append(word_untouched)    #here word_untouched must be appended as the "." and "," were removed otherwise
                else:
                    if post_fix != "": text_line_processed.append(post_fix) #irrelevant, as probably removed within tokenization anyway, but just to be sure
            text_processed.append(text_line_processed)
        text_processed = [" ".join(text_line) for text_line in text_processed]
        #print("text_processed[0]", text_processed[0])
        return text_processed
    
    def load_txt_gt(add):
        txt_gt = pd.read_csv(add)
        class_names = txt_gt.columns[2:].tolist() #header names
        txt_gt["list"] = txt_gt[txt_gt.columns[2:]].values.tolist()
        if config.preprocess_text:
            txt_gt["report"] = preprocess_text(txt_gt["report"], class_names)
        txt_gt = txt_gt[["id", "report", "list"]].copy()
        return txt_gt

    if config.server: monai_dir = "/home/christian/data/TransCheX/monai_data/monai_data" #Server
    else: monai_dir = "/media/christian/Daten1/christian/PhD/PhD_BigData/Projekte/TransCheX/monai_data/monai_data" 
    parent_dir = f"{monai_dir}/dataset_proc/images/"
    train_txt_gt = load_txt_gt(f"{monai_dir}/dataset_proc/train.csv")
    val_txt_gt = load_txt_gt(f"{monai_dir}/dataset_proc/validation.csv")
    test_txt_gt = load_txt_gt(f"{monai_dir}/dataset_proc/test.csv")

    #set max length of dataset for training:
    print("train_txt_gt.shape all", train_txt_gt.shape)
    train_txt_gt = train_txt_gt[:config.train.num_items] if config.train.num_items > 0 and config.train.num_items < len(train_txt_gt) else train_txt_gt
    print("train_txt_gt.shape reduced", train_txt_gt.shape)

    #set max length of dataset for validation:
    print("val_txt_gt.shape all", val_txt_gt.shape)
    val_txt_gt = val_txt_gt[:config.val.num_items] if config.val.num_items > 0 and config.val.num_items < len(val_txt_gt) else val_txt_gt
    print("val_txt_gt.shape reduced", val_txt_gt.shape)

    #set max length of dataset for validation:
    print("test_txt_gt.shape all", test_txt_gt.shape)
    test_txt_gt = test_txt_gt[:config.test.num_items] if config.test.num_items > 0 and config.test.num_items < len(test_txt_gt) else test_txt_gt
    print("test_txt_gt.shape reduced", test_txt_gt.shape)
    
    #select llama_path
    if config.server: llama_path = config.net.llama_path_server
    else: llama_path = config.net.llama_path_local

    if config.net.tokenizer == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)  #TODO: Note: very important to set do_lower_case=True,
                                    # otherwise words at the beginning of a sentence will be set to [UNK] token as they are not in the vocabulary
    else:
        tokenizer = LLaMATokenizer(model_path= os.path.join(llama_path, "tokenizer.model"))

    training_set = MultiModalDataset(train_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    train_params = {
        "batch_size": 1, #set to 1 here! (due to memory requirement)
        "shuffle": config.train.dataload.shuffle,
        "num_workers": config.train.dataload.num_workers,
        "pin_memory": True,
    }
    training_loader = DataLoader(training_set, **train_params)
    valid_set = MultiModalDataset(val_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    test_set = MultiModalDataset(test_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    valid_params = {
        "batch_size": 1, #set to 1 for validation and testing here
        "shuffle": config.val.dataload.shuffle, 
        "num_workers": config.val.dataload.num_workers, 
        "pin_memory": True
    }
    val_loader = DataLoader(valid_set, **valid_params)
    test_loader = DataLoader(test_set, **valid_params)

    print("-------------------------------------------------------")
    print("Length of training dataset: ", len(training_set))
    print("Length of validation dataset: ", len(valid_set))
    print("Length of testing dataset: ", len(test_set))
    print("-------------------------------------------------------")


    # Define the model
    def Net(name: str):
        if name == "ViTLLaMAII":
            model = ImageTextNet( #is ViTLLaMAII
                in_channels=config.net.in_channels,
                img_size=config.image_size,
                num_classes=config.net.num_classes,
                patch_size=config.net.patch_size,
                num_text_layers=config.net.num_text_layers,
                num_vision_layers=config.net.num_vision_layers,
                num_cross_attention_layers=config.net.num_cross_attention_layers,
                num_pre_activation_layers = config.net.num_pre_activation_layers,
                num_pre_activation_layers_cross = config.net.num_pre_activation_layers_cross,
                num_attention_heads_text=config.net.num_attention_heads_text,
                num_attention_heads_vision=config.net.num_attention_heads_vision,
                spatial_dims=config.net.spatial_dims,
                drop_out=config.net.drop_out,
                text_only=config.net.text_only,
                vision_only=config.net.vision_only,
                serial_pipeline=config.net.serial_pipeline,
                #--------------additional variables for llama------------------#
                llama_path = llama_path,
                language_model = config.net.language_model,
                text_max_seq_len = config.text_max_seq_length,
                vocab_size = config.net.vocab_size, #used for num_embeddings
                dim=config.net.dim,
                multiple_of=config.net.multiple_of,
                #--------------additional variables for vit------------------#
                vit_path = config.net.vit_path_server,
                hidden_size_vision = config.net.hidden_size_vision,
                intermediate_size_vision = config.net.intermediate_size_vision,
            ).to(config.device)

        elif name == "ResNetLLaMAII":
            model = ResNetLLaMAII(
                in_channels=config.net.in_channels,
                patch_size=config.net.patch_size, #for embedding for cross transformer
                hidden_size_vision = config.net.hidden_size_vision, # --||--
                img_size=config.image_size,
                num_classes=config.net.num_classes,
                num_text_layers=config.net.num_text_layers,
                num_cross_attention_layers=config.net.num_cross_attention_layers,
                num_pre_activation_layers = config.net.num_pre_activation_layers,
                num_pre_activation_layers_cross = config.net.num_pre_activation_layers_cross,
                num_attention_heads_text=config.net.num_attention_heads_text,
                spatial_dims=config.net.spatial_dims,
                drop_out=config.net.drop_out,
                attention_probs_dropout_prob = config.net.drop_out,
                text_only=config.net.text_only,
                vision_only=config.net.vision_only,
                serial_pipeline=config.net.serial_pipeline,
                #--------------additional variables for llama------------------#
                llama_path = llama_path,
                language_model = config.net.language_model,
                text_max_seq_len = config.text_max_seq_length,
                vocab_size = config.net.vocab_size, #used for num_embeddings
                dim=config.net.dim,
                multiple_of=config.net.multiple_of,
                #--------------additional variables for ResNet------------------#
                conv1_t_size=config.net.conv1_t_size,
                conv1_t_stride=config.net.conv1_t_stride,
                pretrained_vision_net=config.net.pretrained_vision_net,
                vision_act = None,
                #vision_model_path = None,
            ).to(config.device)

        else:
            raise ValueError(f"Model {name} not supported")
        return model

    model = Net(config.net.model_name)

    #----------- Load model with peft -----------#
    task_type = None

    peft_config = LoraConfig(
        task_type=task_type, inference_mode=False, r=config.lora.r, 
        lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, 
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, peft_config)


    #----------- Check best model output with the input image and label -----------#
    # Load the pretrained checkpoint first
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=config.device)["state_dict"],strict=False)
    net_dict = torch.load(os.path.join(lora_model_dir, lora_model_name))
    model.load_state_dict(net_dict["state_dict"], strict=False) #only in .ipynb file needed
    model.eval()
    print(model)

    class Diseases(Enum): 
        Atelectasis =                   [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Cardiomegaly =                  [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Consolidation =                 [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Edema =                         [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Enlarged_Cardiomediastinum =    [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Fracture =                      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Lung_Lesion =                   [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Lung_Opacity =                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        No_Finding =                    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
        Pleural_Effusion =              [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
        Pleural_Other =                 [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
        Pneumonia =                     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
        Pneumothorax =                  [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
        Support_Devices =               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
        not_predicted =                 [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]


    #------------------------------------------INTERPRETABILITY------------------------------------------#
    print("\n\nStarting with Interpretability ...")
    """
    Use GradCAM and occlusion sensitivity for network interpretability.

    The occlusion sensitivity returns two images: the sensitivity image and the most probable class.

        Sensitivity image -- how the probability of an inferred class changes as the corresponding part of the image is occluded.
            Big decreases in the probability imply that that region was important in inferring the given class
            The output is the same as the input, with an extra dimension of size N appended. Here, N is the number of inferred classes. 
            To then see the sensitivity image of the class we're interested (maybe the true class, maybe the predcited class, maybe anything else), we simply do im[...,i].
        Most probable class -- if that part of the image is covered up, does the predicted class change, and if so, to what?
    """

    """

    ---Interpretability
    (Describtion from MONAI (https://github.com/Project-MONAI/tutorials/blob/main/modules/interpretability/covid_classification.ipynb))

    Now we compare our different saliency methods. Initially, the resulting can be tricky to decipher.

    ---Occlusion sensitivity

    With occlusion sensitivity we iteratively block off part of the image and then we record the changes
    in certainty of the inferred class. This means that for instances where the network correctly infers the image type, 
    we expect the certainty to drop as we occlude important parts of the image. Hence, for correct inference, blue parts 
    of the image imply importance.

    This is also true when the network incorrectly infers the image; blue areas were important in inferring the given class.

    ---GradCAM

    The user chooses a layer of the network that interests them and the gradient is calculated at this point. 
    The chosen layer is typically towards the bottom of the network, as all the features have hopefully been extracted 
    by this point. The images have been downsampled many times, and so the resulting images are linearly upsampled to match 
    the size of the input image. As with occlusion sensitivity, blue parts of the image imply importance in the decision 
    making process.
    """

    #check if some examles are already computed (if so, skip them), herein see output_dir/interpretability_information.txt file
    items_processed = []
    if os.path.exists(f"{output_dir}/interpretability_information.txt"):
        with open(f"{output_dir}/interpretability_information.txt", "r") as f:
            #check for lines starting with CXR..., then append this line as it is the image_name
            for line in f:
                if line.startswith("CXR"):
                    items_processed.append(line.strip())

    #num_patches computation:
    num_patches_height = config.image_size[0] // config.net.patch_size[0]
    num_patches_width = config.image_size[1] // config.net.patch_size[1]

    #check if cls token available ! #TODO: BEWARE of that
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

    def reshape_transform_cross(tensor, height=num_patches_height, width=num_patches_width):
        """
        Reshape the tensor to the shape of the patches.
        Input ist img and text.
        """
        print("in_shape: ", tensor.shape)
        result = tensor[:, :, :, :, :, :].reshape(tensor.size(0),
            height, width, tensor.size(2), tensor.size(3), tensor.size(4), tensor.size(5))
        #Bring the channels to the first dimension,
        #like in CNNs
        print(result.shape)
        result = result.transpose(2,3).transpose(1,2)
        print("out_shape: ", result.shape)
        return result

    reshape_function = reshape_transform if config.net.model_name == "ViTLLaMAII" else None

    #interpretability setup
    for name, _ in model.named_modules(): print(name)
    #select a proper layer for gradcam
    if config.net.model_name == "ViTLLaMAII":
        target_layers_vision = [model.base_model.model.multimodal.vision_net.blocks[-1].norm1] #TODO: choose any layer before the final attention block, because it must be a list (even if only one layer) in order to be iterable
    else: #assuming ResNetLLaMAII
        target_layers_vision = [model.base_model.model.multimodal.vision_net.layer4[-1].bn2] #= model.multimodal.vision_net.layer4.2.bn2
    target_layers_text = [model.base_model.model.multimodal.text_net.text_encoder[-1].attention_norm] #TODO: choose any layer before the final attention block, because it must be a list (even if only one layer) in order to be iterable
    target_layers_cross = [model.base_model.model.multimodal.cross_attention_encoder[-1].attention_norm] #TODO: check for functionality
    print("target_layers_vision", target_layers_vision)
    print("target_layers_text", target_layers_text)

    #check if target_layers have "requires_grad" set to True
    print("Checking if target_layers have 'requires_grad' set to True: Not implemented yet")

    gradcam = GradCAM(           #only works for 2D images
            model=model, target_layers=target_layers_vision, 
            use_cuda = True, reshape_transform=reshape_function,
    )

    gradcam_Text = GradCAMText(
            model=model, target_layers=target_layers_text,
            use_cuda=True, reshape_transform=lambda x : x, #TODO: check if more parameters needed
            color_map='turbo', map_colors_to_min_max=True #TODO: change to True...
    )       #for text no reshaping is needed

    gradcam_cross = GradCAMCross(
            model=model, target_layers=target_layers_cross,
            use_cuda=True, reshape_transform=reshape_transform_cross,
            color_map='turbo', map_colors_to_min_max=True,
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

    if config.net.tokenizer == "llama":
        if config.occlusion.text_mask == "mean" or config.occlusion.text_mask is None:
            print("Taking the MEAN for the text mask.")
            mask = None #hence done in occ_sensText __call__() function for each item
            mask_enc = None
        else:
            mask = config.occlusion.text_mask
            mask_enc = tokenizer.encode(mask, bos=False, eos=False)[0]
    else: #bert
        if config.occlusion.text_mask == "mean" or config.occlusion.text_mask is None:
            print("Taking the MEAN for the text mask.")
            mask = None
            mask_enc = None
        else: 
            mask = config.occlusion.text_mask
            mask_enc = tokenizer.convert_tokens_to_ids(mask)[0] #bert
    occ_sensText = OcclusionSensitivityText(
        nn_module=model, replace_token=mask_enc, color_map='turbo',
        map_colors_to_min_max=True, mask_words=True,
    )

    def get_targets_from_probs(probabilities):
        #print("probabilities.shape", probabilities.shape)
        #print("p in probabilities[0]", probabilities[0])
        targets = [1 if (p > 0.5) else 0 for p in probabilities]
        return targets

    def saliency(net, d, num):
        torch.set_printoptions(linewidth=200)
        ims = []
        titles = []
        log_scales = []
        
        #model computation of output probs
        img = d["images"].to(config.device)
        input_ids = d["ids"].to(config.device)
        segment_ids = d["segment_ids"].to(config.device)
        targets = d["targets"].to(config.device)
        text_tokens = d["tokens"]
        print("text_tokens", text_tokens)
        img_name = d["name"]
        pred_logits = net(input_ids=input_ids, image=img, token_type_ids=segment_ids)
        pred_probs = pred_logits.detach().cpu() #already probality values, as sigmoid is applied at the end of the model
        pred_probs = pred_probs[0]  #dereferenced once
        pred_targets = get_targets_from_probs(pred_probs)
        
        print("pred_probs", pred_probs)
        print("pred_targets", pred_targets)

        #Image (shape = [1,3,256,256]) -> [1,256, 256]
        print("img.shape before reshaping",img.shape)
        img_plot = img[:,0,:,:]
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
        print(input_ids)
        if True:    #do not use GradCAM for vision as not working properly yet ue to reshaping or gradient problems (TODO: solve)
            #res_cam = black image, cam_importance_vision = 0
            res_cam = torch.zeros((1, config.image_size[0], config.image_size[1]))
            cam_importance_vision = np.abs(0.0) #just to convert to numpy float number
        else:
            res_cam, cam_importance_vision = gradcam(input_tensor_text=input_ids, input_tensor_vision=img, targets=None) #if target_category = None -> base_cam.py computes this value itself
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
        b_box = None #b_box=[-1,-1,-1,-1,-1,-1, ...] .. if None, then the whole image is used
        occ_map, occ_most_prob = occ_sens(x=img, x_text=input_ids)#, bbox=bbox) #TODO: FIXME: add bounding box here, and adapt occ_sens function (add text input features)
        #Attention: get first predicted class
        pred_label = np.argmax(pred_targets) #FIXME: problem with multiple class
        print("pred_label", pred_label)
        #= occ_sens_shape torch.Size([1, 14, 224, 224]), where 14 = num_classes
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

        occ_map2, occ_sens_class, n, importance_per_class_vision = occ_sens_img(input_ids, img)
        ims.append(occ_map2)
        titles.append("Occ. sens. (CG)")
        log_scales.append(False)

        ims.append(occ_sens_class)
        titles.append("Occ. sens. class (CG), n=" + str(n)) #n: number of patches that forced model to change the classification output
        log_scales.append(False)

        print("\nOcclusion Sensitivity DONE.")
        print("\nStart Grad CAM on text ...")

        #GradCAM on text
        #text_tokens represent the text input phrases
        #TODO: implement GradCAM on text input and call here
        colorized_text_gradCAM, color_bar_gradCAM, cam_importance_text = gradcam_Text(input_tensor_text=input_ids, 
                                      input_tensor_vision=img, 
                                      targets=None, #overwritten anyway within gradcam_Text 
                                      input_tokens=text_tokens) #if target_category = None -> base_cam.py computes this value itself
        file_cam = f'{output_dir}/{img_name[0]}/{img_name[0]}_GradCAM_text.html'
        if not os.path.isdir(f'{output_dir}/{img_name[0]}'):
            os.mkdir(f'{output_dir}/{img_name[0]}')
        with open(file_cam, 'w') as f:
            f.write(colorized_text_gradCAM)
        #save the color bar as html file
        if num == 0:
            file_cbar_cam = f'{output_dir}/GradCAM_text_cbar.html'
            with open(file_cbar_cam, 'w') as f:
                f.write(color_bar_gradCAM)
        else: file_cbar_cam = None
        
        print(f"\nGrad CAM on text DONE.\nWrote {file_cam}")

        #Occ sens on text
        print("\nStart Occlusion Sensitivity on text ...")
        
        colorized_text_occ_sens_mean, color_bar_occ_sens, \
        colorized_text_occ_sens_max, _, \
        occ_sens_class, n, importance_per_class_text = occ_sensText(x_text=text_tokens, 
                                                x_tokens=input_ids,
                                                x_img=img,
                                                )
        file_occ_sens = f'{output_dir}/{img_name[0]}/{img_name[0]}_OccSens_text.html'
        with open(file_occ_sens, 'w') as f:
            f.write("MEAN<br>")
            f.write(colorized_text_occ_sens_mean)
            #in html style: add two breaks
            f.write("<br><br>MAX<br>")
            f.write(colorized_text_occ_sens_max)
            f.write(f"<br><br>{n} words forced model to change the classification output. <br>")
            f.write(occ_sens_class)
        #save the color bar as html file
        if num == 0:
            file_cbar_occ_sens = f'{output_dir}/OccSens_text_cbar.html'
            with open(file_cbar_occ_sens, 'w') as f:
                f.write(color_bar_occ_sens)
        else: file_cbar_occ_sens = None
        print(f"\nOCC sens on text DONE.\nWrote {file_occ_sens}")
        
        print("img.shape", img.shape)
        print("res_cam.shape", res_cam.shape)
        print("ims", ims)

        print("cam_importance_text.shape", cam_importance_text.shape)
        print(f"cam_importance_text = {cam_importance_text}")
        print("cam_importance_vision.shape", cam_importance_vision.shape)
        print(f"cam_importance_vision = {cam_importance_vision}")

        print("importance_per_class_text shape", importance_per_class_text.shape)
        print("importance_per_class_vision shape", importance_per_class_vision.shape)

        #0cc_sesns
        importance_sum = importance_per_class_vision + importance_per_class_text
        importance_per_class_text = importance_per_class_text / importance_sum
        importance_per_class_vision = importance_per_class_vision / importance_sum

        #cam
        cam_importance_sum = cam_importance_vision + cam_importance_text
        cam_importance_text = cam_importance_text / cam_importance_sum
        cam_importance_vision = cam_importance_vision / cam_importance_sum

        #add to file
        #occ sens
        importance_per_class_text_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_text])
        importance_per_class_vision_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_vision])
        text_to_file += f"Occlusion Sensitivity: \n"
        text_to_file += f"Importance per class (text):            [{importance_per_class_text_str}]\n"
        text_to_file += f"Importance per class (vision):          [{importance_per_class_vision_str}]\n"
        text_to_file += f"Importance mean ratio (text : vision) = {np.round(np.mean(importance_per_class_text),2)} : {np.round(np.mean(importance_per_class_vision),2)}\n"
        #cam
        text_to_file += f"CAM: \n"
        text_to_file += f"Importance mean ratio (text : vision) = {np.round(np.mean(cam_importance_text),2)} : {np.round(np.mean(cam_importance_vision),2)}\n"

        temp_text = ""
        if os.path.exists(f"{output_dir}/interpretability_information.txt") and (num > 0 or len(items_processed) > 0):
            with open(f"{output_dir}/interpretability_information.txt", 'r') as fp:
                temp_text = fp.read()
        with open(f"{output_dir}/interpretability_information.txt", "w") as f:
            f.write(temp_text)
            if temp_text != "": f.write("\n")
            f.write(text_to_file)


        return ims, titles, log_scales, [file_cam, file_cbar_cam], [file_occ_sens, file_cbar_occ_sens], importance_per_class_text, importance_per_class_vision, cam_importance_text, cam_importance_vision

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

    #now iterate over test_loader and compute saliency for each item
    #--------------------------------------------------------------------------------
    num_examples = config.occlusion.num_examples #number of examples to be plotted
    #rand_data = np.random.choice(test_set, replace=False, size=num_examples) #error: but not needed
    #print(rand_data)

    initialized_arrays = False
    for row, d in enumerate(test_loader):   #test_loader, val_loader, training_loader
        print("\nProcessing item...", row)
        print("d[images].shape", d["images"].shape)
        #check if item is already processed, items_processed is a list of image names, that are already computed in a earlier run
        print("d[name][0]", d["name"][0])
        if d["name"][0] in items_processed:
            print(f"Item {d['name'][0]} already processed. Skipping ...")
            continue
        else:
            print(f"Item {d['name'][0]} not yet processed. Processing ...")
        ims, titles, log_scales, colorized_data_cam, colorized_data_occ_sens, \
        importance_text, importance_vision, cam_importance_text, cam_importance_vision = saliency(model, d, row)
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

        #check if arrays are already exists
        if not initialized_arrays:
            importance_per_class_text_all = np.empty((0,importance_text.shape[0]))
            importance_per_class_vision_all = np.empty((0,importance_vision.shape[0]))
            cam_importance_text_all = np.empty(0)
            cam_importance_vision_all = np.empty(0)
            initialized_arrays = True

        importance_per_class_text_all = np.concatenate((importance_per_class_text_all, [importance_text]), axis=0)
        importance_per_class_vision_all = np.concatenate((importance_per_class_vision_all, [importance_vision]), axis=0)
        cam_importance_text_all = np.concatenate((cam_importance_text_all, [cam_importance_text]), axis=0)
        cam_importance_vision_all = np.concatenate((cam_importance_vision_all, [cam_importance_vision]), axis=0)

        if row == (num_examples-1):
            break
    
    #-----------------------------------------------------------------------------------------------#
    #first, check if items_processed is not empty. If so, we have to add Importances of these items to the correct arrays
    if len(items_processed) > 0:
        with open(f"{output_dir}/interpretability_information.txt", "r") as f:
            #search for the image_names in items_processed and add the importances to the correct arrays
            for line in f:
                if line.startswith("CXR"):
                    img_name = line.strip()
                    if img_name in items_processed:
                        #search for the next line with "Importance per class (text):"
                        line = f.readline()
                        while not line.startswith("Importance per class (text):"):
                            line = f.readline()
                        #add the importances to the correct arrays
                        importance_per_class_text = [float(entry) for entry in line.split("[")[1].split("]")[0].split(",")]
                        #search for the next line with "Importance per class (vision):"
                        line = f.readline()
                        while not line.startswith("Importance per class (vision):"):
                            line = f.readline()
                        #add the importances to the correct arrays
                        importance_per_class_vision = [float(entry) for entry in line.split("[")[1].split("]")[0].split(",")]
                        #same for CAM
                        #search for the next line with "CAM:"
                        line = f.readline()
                        while not line.startswith("CAM:"):
                            line = f.readline()
                        #next line is" Importance mean ratio (text : vision) = x : y
                        line = f.readline()
                        cam_importance_text = float(line.split("=")[1].split(":")[0].strip())
                        cam_importance_vision = float(line.split("=")[1].split(":")[1].strip())
                        #check if arrays are already exists (could happen if no items are processed yet)
                        if not initialized_arrays:
                            importance_per_class_text_all = np.empty((0,len(importance_per_class_text)))
                            importance_per_class_vision_all = np.empty((0, len(importance_per_class_vision)))
                            cam_importance_text_all = np.empty(0)
                            cam_importance_vision_all = np.empty(0)
                            initialized_arrays = True
                        #add to arrays
                        importance_per_class_text_all = np.concatenate((importance_per_class_text_all, [importance_per_class_text]), axis=0)
                        importance_per_class_vision_all = np.concatenate((importance_per_class_vision_all, [importance_per_class_vision]), axis=0)
                        cam_importance_text_all = np.concatenate((cam_importance_text_all, [cam_importance_text]), axis=0)
                        cam_importance_vision_all = np.concatenate((cam_importance_vision_all, [cam_importance_vision]), axis=0)

                        
    #OCC sensitivity
    #compute mean importance per class
    importance_per_class_text_mean = np.mean(importance_per_class_text_all, axis=0)
    importance_per_class_vision_mean = np.mean(importance_per_class_vision_all, axis=0)
    importance_text_mean = np.mean(importance_per_class_text_mean)
    importance_vision_mean = np.mean(importance_per_class_vision_mean)
    importance_sum = importance_text_mean + importance_vision_mean
    importance_text_mean = importance_text_mean / importance_sum
    importance_vision_mean = importance_vision_mean / importance_sum
    importance_per_class_text_mean_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_text_mean])
    importance_per_class_vision_mean_str = ", ".join([f"{entry:0.2f}" for entry in importance_per_class_vision_mean])

    #CAM
    cam_text_mean = np.mean(cam_importance_text_all, axis=0)
    cam_vision_mean = np.mean(cam_importance_vision_all, axis=0)
    cam_sum = cam_text_mean + cam_vision_mean
    cam_text_mean = cam_text_mean / cam_sum
    cam_vision_mean = cam_vision_mean / cam_sum

    #save mean importance per class to file
    with open(f"{output_dir}/interpretability_information.txt", "a") as f:
        f.write(f"\n\nOcc_sens:\n")
        f.write(f"Mean importance per class (text):            [{importance_per_class_text_mean_str}]\n")
        f.write(f"Mean importance per class (vision):          [{importance_per_class_vision_mean_str}]\n")
        f.write(f"Mean importance ratio (text : vision) = {np.round(importance_text_mean,2)} : {np.round(importance_vision_mean,2)}\n")
        f.write("\nCAM:\n")
        f.write(f"Mean importance ratio (text : vision) = {np.round(cam_text_mean,2)} : {np.round(cam_vision_mean,2)}\n")
        
    print("Finished with Interpretability ...")
    
    #postprocessing done in python script postprocessing.py
    #-----------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main(sys.argv[1])
