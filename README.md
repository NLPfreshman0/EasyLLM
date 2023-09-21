<div align="center">
  <h1>EasyLLM</h1>
</div>

## 简介
这是一个自用大模型仓库，致力于汇总当前可用的大模型资源，并提供下载链接，同时提供了一些大模型的训练代码和相关工具。
## 内容清单
1. [大模型总结](#1-大模型总结)
2. [环境配置](#2-环境配置)
3. [数据集准备](#3-数据集准备)
4. [训练代码](#4-训练代码)
5. [模型推理](#5-模型推理)
6. [使用工具](#6-使用工具)

## 1. 大模型总结
在这个部分，你可以找到已经收集的各种大模型的列表。对于开源的大模型将提供下载链接。
### 国外大模型
| 序号 | 模型名称 | 机构 | 简介 | 下载地址 |
| --- | --- | --- | ----- | --- |
| 1 | **GPT-4**, **GPT-3.5**, **GPT-3**, **ChatGPT**, **Instruction GPT** | openai | openai的系列大模型，未开源，可通过api访问 | [官网](https://openai.com/) |
| 2 | **LLaMA** | Meta | 首个开源大模型 | [7B](https://huggingface.co/huggyllama/llama-7b)\|[13B](https://huggingface.co/huggyllama/llama-13b)\|[30B](https://huggingface.co/huggyllama/llama-30b)\|[65B](https://huggingface.co/huggyllama/llama-65b) |
| 3 | **LLaMA2** | Meta | 与LLaMA相比，训练数据提高40%，上下文长度翻倍 | [7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)\|[13B](https://huggingface.co/meta-llama/Llama-2-13b)\|[70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)<br>[chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)\|[chat-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)\|[chat-70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)|
| 4 | **NLLB** | meta | 支持200种语言互译 | [distill-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)\|[distill-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)\|[1.3B](https://huggingface.co/facebook/nllb-200-1.3B)\|[3.3B](https://huggingface.co/facebook/nllb-200-3.3B)\|[54B](https://huggingface.co/facebook/nllb-moe-54.5B) |
| 5 | **BLOOM** | bigscience | 首个参数量超过100B的开源大模型 | [560M](https://huggingface.co/bigscience/bloom-560m)\|[1.1B](https://huggingface.co/bigscience/bloomz-1b1)\|[1.7B](https://huggingface.co/bigscience/bloom-1b7)\|[3B](https://huggingface.co/bigscience/bloom-3b)\|[7.1B](https://huggingface.co/bigscience/bloom-7b1)\|[176B](https://huggingface.co/bigscience/bloom) |
| 5 | **Falcon** | TIIuae | 基于BLOOM模型架构的改进，使用了multi-query和flash-attention技术，更高质量的数据集 | [7B](https://huggingface.co/tiiuae/falcon-7b)\|[instruct-7B](https://huggingface.co/tiiuae/falcon-7b-instruct)\|[40B](https://huggingface.co/tiiuae/falcon-40b)\|[instruct-40B](https://huggingface.co/tiiuae/falcon-40b-instruct)\|[180B](https://huggingface.co/tiiuae/falcon-180B)\|[chat-180B](https://huggingface.co/tiiuae/falcon-180B-chat) |
### 国内大模型
| 序号 | 模型名称 | 机构 | 简介 | 下载地址 |
| --- | --- | --- | ----- | --- |
| 1 | **ChatGLM** | 清华&智谱 | 支持中英双语问答的对话语言模型，并针对中文进行了优化,仅6B开源 | [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) |
| 2 | **ChatGLM2** | 清华&智谱 | 具有更长的上下文以及更高效的推理速度， 仅6B开源 | [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) |
| 3 | 模型C | 机构C |  | [下载链接](https://example.com/modelC) |
| 4 | 模型D | 机构D |  | [下载链接](https://example.com/modelD) |
| 5 | 模型E | 机构E |  | [下载链接](https://example.com/modelE) |

## 2. 环境配置
python >= 3.9<br>
pytorch >= 2.0<br>
(conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia)<br>
bitsandbytes==0.39.0 #QLoRA量化<br>
deepspeed            #加速训练框架<br>
详细环境配置见**requirement.txt**

## 3. 数据集准备
本仓库的数据集均为csv格式文件，根据不同的训练方式，文件格式略有不同
### 1.预训练

### 2.sft
- 在整个样本上计算loss，仅需一列数据：<br>
```
格式："<s>Human: +问题+\n</s><s>Assistant: +答案+</s>" <br>
例如："<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>"<br>
```
- 仅在label上计算loss，需处理成两列(列名随意)：<br>
```
格式：第一列："<s>Human: +问题+\n</s><s>Assistant: "  第二列："答案+</s>"<br>
例如：第一列："<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: "  第二列："因为地球是目前为止唯一已知存在生命的行星。</s>"<br>
```
文件样例可见data/train_sft.csv

### 3.分类任务
两列数据，第一列为要分类的文本，第二列为标签 [text, label]

## 4. 训练代码
### 1. 预训练
- 预训练脚本见train/pretrain/pretrain.sh,预训练代码为train/pretrain/pretrain_clm.py<br>
- deepspeed加速配置文件，单卡训练使用train/pretrain/ds_config_zero2.json，多卡训练使用train/pretrain/ds_config_zero3.json<br>
- 没有足够的资源进行预训练，该脚本暂未亲自测试

### 2. sft LoRA微调
脚本为train/sft/finetune_lora.sh:
```
output_model=save_folder                                                    #设置保存模型的路径
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then                                   
    mkdir ${output_model}
fi
cp ./finetune.sh ${output_model}
deepspeed --include localhost:0 --master_port 29505  finetune_clm_lora.py \ #设置使用的显卡编号与端口
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \                    #设置训练的基模型路径
    --train_files ../../data/train_sft.csv \                                #训练集路径，支持输入多个文件
                ../../data/train_sft_sharegpt.csv \
    --validation_files  ../../data/dev_sft.csv \                            #验证集路径
                         ../../data/dev_sft_sharegpt.csv \
    --per_device_train_batch_size 1 \                                       #每张卡上训练的batch_size
    --per_device_eval_batch_size 1 \                                        #每张卡上验证的batch_size
    --do_train \                                                            #是否训练
    --do_eval \                                                             #是否验证
    --use_fast_tokenizer false \                                            
    --output_dir ${output_model} \                                          #保存模型的路径，这里不用修改
    --evaluation_strategy  steps \                                          #验证的策略，默认为steps
    --max_eval_samples 800 \                                                #验证集最大样本数
    --learning_rate 1e-4 \                                                  #学习率
    --gradient_accumulation_steps 8 \                                       #梯度累计的steps
    --num_train_epochs 10 \                                                 #训练的epoch
    --warmup_steps 400 \                                                    #学习率预热步数
    --load_in_bits 4 \                                                      #使用的bit数，默认4bit
    --lora_r 8 \                                                            #LoRA的秩
    --lora_alpha 32 \                                                       #LoRA模型的权重
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \ #LoRA作用的模块
    --logging_dir ${output_model}/logs \                                    #log保存的目录，用于tensorboard可视化，不用修改
    --logging_strategy steps \                                              #保存日志的策略
    --logging_steps 10 \                                                    #保存日志的steps
    --save_strategy steps \                                                 #保存模型的策略
    --preprocessing_num_workers 10 \                                        #处理数据集的线程数
    --save_steps 20 \                                                       #保存模型的steps
    --eval_steps 20 \                                                       #验证的steps
    --save_total_limit 2000 \                                               #保存的最大模型数
    --seed 42 \                                                             #随机种子
    --disable_tqdm false \                                                  #不显示tqdm，默认为False
    --ddp_find_unused_parameters false \                                    
    --block_size 2048 \                                                     #最大的tokenizer长度，受限于模型
    --report_to tensorboard \                                               
    --overwrite_output_dir \                                                
    --deepspeed ds_config_zero2.json \                                      #deepspeed配置文件
    --ignore_data_skip true \                                               
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \          #是否从checkpoint开始训练，从checkpoint训练需指定路径
```
### 3.分类任务lora微调
- 脚本为train/sft/finetune_cls.sh，参数与sft lora微调类似，会额外保存最后的分类层权重score_layer_weights.bin
- 模型加载可参考train/sft/test_cls.py
## 5. 模型推理
### 1.加载模型时合并lora权重
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
finetune_model_path=''                        #lora权重所在路径
config = PeftConfig.from_pretrained(finetune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```
### 2.先合并LoRA权重，使用脚本train/merge/merge.sh，再直接加载合并后的模型进行推理
合并权重
```
CUDA_VISIBLE_DEVICES=0 python merge_peft_adapter.py \
    --adapter_model_name /checkpoint-2200 \           #lora的checkpoint所在目录
    --output_name checkpoint-2200_merge \             #合并之后模型的存储位置
    --load8bit false \                                #是否执行8bit量化
    --tokenizer_fast false                            #llama模型只能为false
```
加载模型
```
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
后面的代码同1
```
对于需要多次使用的模型，先合并可加快之后加载的速度
## 6. 使用工具
在这个部分，你会找到一些用于使用和部署大模型的工具和实用程序。
### 1.构建webui,可以供本地或互联网访问
python chat_gradio.py --model_name_or_path=model_path --share #share参数允许互联网访问
![image](https://github.com/NLPfreshman0/EasyLLM/assets/55648342/b919372f-d528-45f5-a991-1d8112a4a114)
### 2.构建api，允许本地访问
使用fastapi 首先pip install fastapi uvicorn
#### 1.启动服务
```
python acceleate_server.py \
--model_path path \
--gpus "0" \
--infer_dtype "int8" \
```
参数说明：
- model_path 模型的本地路径
- gpus 使用的显卡编号，类似"0"、 "0,1"
- infer_dtype 模型加载后的参数数据类型，可以是 int4, int8, float16，默认为int8
#### 2.启动测试客户端，可自行修改此脚本
```
python acceleate_client.py
```






