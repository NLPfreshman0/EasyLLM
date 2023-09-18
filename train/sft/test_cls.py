import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig,PeftModel
import pandas as pd
from tqdm import tqdm
import os

#加载base以及lora模型
fintune_model_path = ''
config = PeftConfig.from_pretrained(fintune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
model =  AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True,num_labels=3)
model = PeftModel.from_pretrained(model, fintune_model_path)

#加载最后的分类层
classification_layer_path = os.path.join(fintune_model_path, score_layer_weights.bin)
classification_layer_params = torch.load(classification_layer_path)
model.score.load_state_dict(classification_layer_params)
model.eval()


