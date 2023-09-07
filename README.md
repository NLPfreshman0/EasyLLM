<div align="center">
  <h1>EasyLLM</h1>
</div>

## 简介
这是一个自用大模型仓库，致力于汇总当前可用的大模型资源，并提供下载链接，同时提供了一些大模型的训练代码和相关工具。
## 内容清单
1. [大模型总结](#1-大模型总结)
2. [环境配置](#2-环境配置)
3. [训练代码](#2-训练代码)
4. [使用工具](#3-使用工具)

## 1. 大模型总结
在这个部分，你可以找到已经收集的各种大模型的列表。对于开源的大模型将提供下载链接。
### 国外大模型
| 序号 | 模型名称 | 机构 | 简介 | 下载地址 |
| --- | --- | --- | ----- | --- |
| 1 | **GPT-4**, **GPT-3.5**, **GPT-3**, **ChatGPT**, **Instruction GPT** | openai | openai的系列大模型，未开源，可通过api访问 | [官网](https://openai.com/) |
| 2 | **LLaMA** | Meta | 首个开源大模型 | [7B](https://huggingface.co/huggyllama/llama-7b)\|[13B](https://huggingface.co/huggyllama/llama-13b)\|[30B](https://huggingface.co/huggyllama/llama-30b)\|[65B](https://huggingface.co/huggyllama/llama-65b) |
| 3 | **LLaMA2** | Meta | 与LLaMA相比，训练数据提高40%，上下文长度翻倍 | [7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)\|[13B](https://huggingface.co/meta-llama/Llama-2-13b)\|[70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)<br>[Chat7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)\|[Chat13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)\|[Chat70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)|
| 4 | 模型D | 机构D |  | [下载链接](https://example.com/modelD) |
| 5 | 模型E | 机构E |  | [下载链接](https://example.com/modelE) |
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

### 2.微调

### 3.sft
- 在整个样本上计算loss，仅需一列数据：<br>
'''
格式：\"<s>Human: +问题+\n</s><s>Assistant: +答案+</s>\" <br>
例如：\"<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>\"<br>
'''
- 仅在label上计算loss，需处理成两列：<br>
'''
格式：第一列：\"<s>Human: +问题+\n</s><s>Assistant: \"  第二列："答案+</s>\"<br>
例如：第一列：\"<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: \"  第二列：\"因为地球是目前为止唯一已知存在生命的行星。</s>\"<br>
'''
文件样例可见data/train_sft.csv

## 3. 训练代码
### 1. 预训练
- 预训练脚本见train/pretrain/pretrain.sh,预训练代码为train/pretrain/pretrain_clm.py<br>
- deepspeed加速配置文件，单卡训练使用train/pretrain/ds_config_zero2.json，多卡训练使用train/pretrain/ds_config_zero3.json<br>
- 没有足够的资源进行预训练，该脚本暂未亲自测试


## 4. 使用工具
在这个部分，你会找到一些用于使用和部署大模型的工具和实用程序。这些工具可能包括：
- 模型推理脚本
- 文本生成示例
- 模型可视化工具
- 部署到生产环境的指南


