import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig,PeftModel
import pandas as pd
from tqdm import tqdm

fintune_model_path = '/data/zhangdacao/AtomGPT/save/llama13b-snli-lora_1/checkpoint-4400'
config = PeftConfig.from_pretrained(fintune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
model =  AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True,num_labels=3)

model = PeftModel.from_pretrained(model, fintune_model_path)
classification_layer_path = '/data/zhangdacao/AtomGPT/save/llama13b-snli-lora_1/checkpoint-4400/score_layer_weights.bin'
classification_layer_params = torch.load(classification_layer_path)
model.score.load_state_dict(classification_layer_params)


def test(path):
    true = 0
    all = 0
    sf = torch.nn.Softmax(dim=-1)
    dev = pd.read_csv(path)
    for index, row in tqdm(dev.iterrows()):
        text = row['text']
        label = row['label']
        inputs = tokenizer(text,truncation=True,max_length=256,padding="max_length",return_tensors='pt')
        output = model(**inputs)
        logits = sf(output.logits.float())
        pred = torch.argmax(logits)
        all += 1
        if pred == label:
            true += 1
    print(true, all, true/all)
 
print('validation:')        
test('/data/zhangdacao/AtomGPT/AtomGPT-main/data/snli/validation.csv')
print('test:')
test('/data/zhangdacao/AtomGPT/AtomGPT-main/data/snli/test.csv')

