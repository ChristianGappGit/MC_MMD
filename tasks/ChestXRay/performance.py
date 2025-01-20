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
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from monai.config import print_config
from monai.utils import set_determinism
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)

from peft import get_peft_model, LoraConfig
from monai.networks.utils import eval_mode
from transformers import BertTokenizer
import loralib as lora
#from ImageTextNet_Peft import ImageTextNet_Peft #nn.Linear Layers used
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

def main(performance_dir: str):

    #----------- Download and pre-process dataset -----------#

    #DONE
    datadir = "./monai_data"
    #if not os.path.exists(datadir):
    #    raise ValueError("Please download the dataset from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data and place it in the monai_data folder")


    #----------- Print Configurations  -----------#
    torch.backends.cudnn.benchmark = True

    print_config()

    #----------- Load Configuration File -----------#
    perform_dir = performance_dir
    print("perform_dir", perform_dir)

    config_file = f"{perform_dir}/default_llama.yaml"
    config = load_config(config_file)
    print(config)

    model_dir = f"{perform_dir}/best_model" #must exist
    model_name = "transchex.pt" #must exist
    lora_model_dir = f"{perform_dir}/lora_model" #must exist
    lora_model_name = "transchex_lora.pt" #must exist
    if config.performance.text_available and config.performance.vision_available:
        modality_name = "text_vision"
    elif config.performance.text_available:
        modality_name = "text"
    elif config.performance.vision_available:
        modality_name = "vision"
    else:
        print("ERROR: No input data available. Modify config file.")
        sys.exit(1)
    output_dir = f"{perform_dir}/output"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = f"{output_dir}/performance_{modality_name}"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

        #----------- Setup dataloaders and transforms for training/validation/testing -----------#
    class MultiModalDataset(Dataset): #Cachedataset possible, but not urgently needed for 2D images
        def __init__(self, dataframe, tokenizer, tokenizer_type, parent_dir, max_seq_length=config.text_max_seq_length):
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

        def encode_features_llama(self, sent, max_seq_length):
            #llama tokenization
            tokens = []
            return_tokens = []
            #split sent to vec of words
            sent = sent.split()
            #print(sent)
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
            return_tokens = [item for sublist in return_tokens for item in sublist]
                #not needed as words can than be reconstructed from tokens easier 
 

            assert(len(tokens) == len(return_tokens))

            #now tokens are encoded properly
            if len(tokens) > max_seq_length: #-2 when adding cls and sep tokens
                tokens = tokens[: (max_seq_length)] #-2
                return_tokens = return_tokens[: (max_seq_length)] #-2
            #tokens = tokens #[0] + tokens + [2] TODO: check if correct (and add: tokens = ["[CLS]"] + tokens + ["[SEP]"]) ??
            input_ids = tokens

            #print("input_ids", input_ids)
            #print("type_input_ids", type(input_ids))
            #print("type_input_ids[0]", type(input_ids[0]))  #correct

            segment_ids = [0] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                segment_ids.append(0)
                return_tokens.append("**NULL**")    #CHECK if correct "**NULL**"
            #print("tokens", tokens)
            assert len(input_ids) == max_seq_length
            assert len(segment_ids) == max_seq_length
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
            targets = torch.tensor(self.targets[index], dtype=torch.float) #TODO: changed to half precision
            return {
                "ids": input_ids,
                "segment_ids": segment_ids,
                "name": name,
                "targets": targets,
                "images": images,
                "tokens": tokens, #not needed here (but for CAM in interpretability.py)
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

    if config.server: llama_path = config.net.llama_path_server
    else: llama_path = config.net.llama_path_local
    
    if config.net.tokenizer == "bert":
        print("Use tokenizer: bert")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)  #TODO: Note: very important to set do_lower_case=True,
                                    # otherwise words at the beginning of a sentence will be set to [UNK] token as they are not in the vocabulary
    else:
        print("Use tokenizer: llama")
        tokenizer = LLaMATokenizer(model_path= os.path.join(llama_path, "tokenizer.model"))

    training_set = MultiModalDataset(train_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    train_params = {
        "batch_size": config.train.dataload.batch_size,
        "shuffle": config.train.dataload.shuffle,
        "num_workers": config.train.dataload.num_workers,
        "pin_memory": True,
    }
    training_loader = DataLoader(training_set, **train_params)
    valid_set = MultiModalDataset(val_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    test_set = MultiModalDataset(test_txt_gt, tokenizer, config.net.tokenizer, parent_dir)
    valid_params = {"batch_size": config.val.dataload.batch_size, "shuffle": config.val.dataload.shuffle, "num_workers": config.val.dataload.num_workers, "pin_memory": True}
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

    loss_bce = torch.nn.BCELoss().to(config.device)

    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def save_ckp_lora(state, checkpoint_dir):
        torch.save(lora.lora_state_dict(model), checkpoint_dir)

    def compute_AUCs(gt, pred, num_classes=14):
        with torch.no_grad():
            AUROCs = []
            gt_np = gt
            pred_np = pred
            for i in range(num_classes):
                AUROCs.append(roc_auc_score(gt_np[:, i].tolist(), pred_np[:, i].tolist()))
        return AUROCs

    def compute_ACC(gt, pred):
        """
        gt: ground truth (output probabilities),
        pred: prediction (targets (same size as gt))"
        """
        #worst ACC = 0: 14*(0)/14 = 0
        #best ACC = 1: 14*(1)/14 = 1
        
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()
        
        with torch.no_grad():
            acc = np.zeros(len(pred[0]))
            #print("pred", pred)
            #print("acc", acc)
            for pred_item, gt_item in zip(pred, gt):
                #print("pred_item",pred_item)
                acc_val = []
                for pred_entry, gt_entry in zip(pred_item, gt_item):
                    #print("pred_entry", pred_entry)
                    pred_entry = 0 if pred_entry < 0.5 else 1
                    entry = 1 if pred_entry == gt_entry else 0
                    acc_val.append(entry) #shape=[1x14]
                acc += acc_val #shape=[1x14]
                #print("acc_vec", acc)
            #print("acc_vec", acc)
            acc /= len(pred)
            #print("acc", acc)
        return acc

    def validation(testing_loader, val: bool = True):
        model.eval()
        #targets_in = np.zeros((len(testing_loader)*config.val.dataload.batch_size, 14))    #TODO: delete, was needed for np list
        #preds_cls = np.zeros((len(testing_loader)*config.val.dataload.batch_size, 14))
        targets_in = torch.tensor([], dtype=torch.float32, device=config.device)
        preds_cls = torch.tensor([], dtype=torch.float32, device=config.device)
        val_loss = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                input_ids = data["ids"].to(config.device) if config.performance.text_available else None
                segment_ids = data["segment_ids"].to(config.device)
                img = data["images"].to(config.device) if config.performance.vision_available else None
                targets = data["targets"].to(config.device)
                #print("input data")
                #print("input_ids", input_ids)
                #print("segment_ids", segment_ids)
                #print("img", img)
                #print("targets", targets)
                logits_lang = model(
                    input_ids=input_ids, image=img, token_type_ids=segment_ids
                )
                prob = logits_lang #was torch.sigmoid(logits_lang) # but wrong, already in model realised
                loss = loss_bce(prob, targets).item()
                #print(prob)
                #print(loss)
                #print("targets_in.shape", targets_in.shape)
                #print("targets.shape", targets.shape)
                #print("targets.detach().cpu().numpy().shape", targets.detach().cpu().numpy().shape)
                #print("targets.shape[0]", targets.shape[0])
                #targets_in[_*config.val.dataload.batch_size : _*config.val.dataload.batch_size+targets.shape[0], :] = targets.detach().cpu().numpy() #targets.shape[0] == batch_size at least for last batch !!! #TODO: delete
                #print("preds_cls.shape", preds_cls.shape)
                #print("prob.shape", prob.shape)
                #preds_cls [_*config.val.dataload.batch_size : _*config.val.dataload.batch_size+prob.shape[0], :] = prob.detach().cpu().numpy()
                targets_in = torch.cat([targets_in, targets], dim=0)
                preds_cls = torch.cat([preds_cls, prob], dim=0)
                val_loss.append(loss)
            auc = compute_AUCs(targets_in, preds_cls, 14)
            acc = compute_ACC(targets_in, preds_cls)
            mean_auc = np.mean(auc)
            mean_acc = np.mean(acc)        
            mean_loss = np.mean(val_loss)
            print(
                "Evaluation Statistics: Mean AUC : {}, Mean ACC {},  Mean Loss : {}".format(
                    mean_auc, mean_acc, mean_loss
                )
            )
        return mean_auc, mean_acc, mean_loss, auc, acc


    #----------- Check best model output with the input image and label -----------#
    # Load the pretrained checkpoint first
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=config.device)["state_dict"],strict=False)
    net_dict = torch.load(os.path.join(lora_model_dir, lora_model_name))
    model.load_state_dict(net_dict["state_dict"], strict=False) #only in .ipynb file needed
    model.eval()
    print(model)
    #print("state_dict", net_dict["state_dict"])

    def AUC_ACC_evaluation(loader, datasetnameprefix = ""):
        val = False #setting val False for all time as no tensorboard writing wished within this function
        with torch.no_grad():
            auc_val, acc_val, loss_val, auc, acc = validation(loader, False)

        Evaluation_statistics = "TESTING STATISTICS:\nMean AUC : {}, Mean Loss : {}\n\nMean test AUC for each class in 14 disease categories\
            :\n\nAtelectasis: {}\nCardiomegaly: {}\nConsolidation: {}\nEdema: \
            {}\nEnlarged-Cardiomediastinum: {}\nFracture: {}\nLung-Lesion: {}\nLung-Opacity: \
            {}\nNo-Finding: {}\nPleural-Effusion: {}\nPleural_Other: {}\nPneumonia: \
            {}\nPneumothorax: {}\nSupport-Devices: {}".format(
                auc_val, loss_val,
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
        Evaluation_statistics += "\n\nTESTING STATISTICS:\nMean ACC : {}, Mean Loss : {}\n\nMean test ACC for each class in 14 disease categories\
            :\n\nAtelectasis: {}\nCardiomegaly: {}\nConsolidation: {}\nEdema: \
            {}\nEnlarged-Cardiomediastinum: {}\nFracture: {}\nLung-Lesion: {}\nLung-Opacity: \
            {}\nNo-Finding: {}\nPleural-Effusion: {}\nPleural_Other: {}\nPneumonia: \
            {}\nPneumothorax: {}\nSupport-Devices: {}".format(
                acc_val, loss_val,
                acc[0],
                acc[1],
                acc[2],
                acc[3],
                acc[4],
                acc[5],
                acc[6],
                acc[7],
                acc[8],
                acc[9],
                acc[10],
                acc[11],
                acc[12],
                acc[13],
            )
        print(Evaluation_statistics)
        tempdata = ""
        if os.path.exists(f"{output_dir}/evaluation_testing_statistics.txt"):
            with open(f"{output_dir}/evaluation_testing_statistics.txt", 'r') as fp:
                tempdata = fp.read()
        with open(f"{output_dir}/evaluation_testing_statistics.txt", 'w') as fp1:
            fp1.write(tempdata)
            fp1.write("\n\n")
            fp1.write(f"Performance on {datasetnameprefix} dataset:\n")
            fp1.write(Evaluation_statistics)
        return

    AUC_ACC_evaluation(test_loader, "test")
    AUC_ACC_evaluation(val_loader, "val")
    #AUC_ACC_evaluation(training_loader, "train")

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
        not_predicted =                 [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0] #only for not predicted classes

    #--------------------------------------------------------------------------------
    #Performance
    def performance(loader, filenamepostfix: str="val"):
        """ 
        Classification report for a given net and dataset.
        loader = validation or train loader
        """
        model.eval()

        # Evaluate best model
        with eval_mode(model): #or FIXME: with torch.no_grad(): ?
            y_gt = torch.tensor([], dtype=torch.float32, device=config.device)
            y_pred = torch.tensor([], dtype=torch.float32, device=config.device)
            for data in loader:
                if not config.performance.text_available and not config.performance.vision_available:
                    print("ERROR: No input data available. Modify config file.")
                    sys.exit(1)
                input_ids = data["ids"].to(config.device) if config.performance.text_available else None
                segment_ids = data["segment_ids"].to(config.device)
                img = data["images"].to(config.device) if config.performance.vision_available else None
                targets = data["targets"].to(config.device)
                outputs = model(
                    input_ids=input_ids, image=img, token_type_ids=segment_ids
                )
                #print("outputs",outputs)
                #print("targets", targets)
                #print("")
                #As multiple classes can be predicted, we need to find the indeces of the classes with the highest probability
                list_indices_gt = [[i for i in range(0,len(vec)) if vec[i] == torch.max(vec)] for vec in targets]
                list_indices_pred = [[i for i in range(0,len(vec)) if vec[i] == torch.max(vec)] for vec in outputs]
                #y_gt = torch.cat([y_gt, targets.argmax(dim=1)], dim=0)
                #y_pred = torch.cat([y_pred, outputs.argmax(dim=1)], dim=0)
                for i in range(len(list_indices_gt)): #== len(list_indices_pred)
                    while len(list_indices_gt[i]) < len(list_indices_pred[i]):  #will rarely (normally never) happen
                        list_indices_gt[i].append(16) #append wrong gt class
                    while len(list_indices_pred[i]) < len(list_indices_gt[i]):
                        list_indices_pred[i].append(16) #append wrong pred class
                #print("indices_gt", list_indices_gt)
                #print("indices_pred", list_indices_pred)
                #add entries as tensor to y* tensors
                for entry in list_indices_gt:
                    y_gt = torch.cat([y_gt, torch.tensor(entry, dtype=torch.float32, device=config.device)], dim=0)
                for entry in list_indices_pred:
                    y_pred = torch.cat([y_pred, torch.tensor(entry, dtype=torch.float32, device=config.device)], dim=0)

        print("y_gt", y_gt)
        print("y_pred", y_pred)
        # Performance Report
        print(classification_report(
            y_gt.cpu().numpy(),
            y_pred.cpu().numpy(),
            target_names=[d.name for d in Diseases])
        )

        cm = confusion_matrix(
            y_gt.cpu().numpy(),
            y_pred.cpu().numpy(),
            normalize='true',
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[d.name for d in Diseases],
        )
        disp.plot(ax=plt.subplots(1,1,facecolor='white')[1])
        plt.xticks(rotation=45, ha='right')
        plt.savefig(f"{output_dir}/confusion_{filenamepostfix}.png")
        plt.savefig(f"{output_dir}/confusion_{filenamepostfix}.eps")
        return

    #--------------------------------------------------------------------------------

    #Test performance
    performance(test_loader, "test")

    #Validation performance
    performance(val_loader, "val")

    #Training performance
    #performance(training_loader, "train") #most often not needed, uncomment if needed

if __name__ == "__main__":
    main(sys.argv[1])
