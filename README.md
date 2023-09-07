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
python >= 3.9
pytorch >= 2.0 
(conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia)
bitsandbytes==0.39.0 #QLoRA量化
deepspeed            #加速训练框架
git+https://github.com/PanQiWei/AutoGPTQ.git  #量化
其它的环境详见requirement.txt

## 3. 训练代码
### 1. 预训练
这一部分包含了一些大模型的训练代码示例，可以帮助你了解如何从头开始训练大模型或者微调现有模型以适应特定任务。每个示例都会包含：
- 训练数据集的准备方法
- 模型训练的步骤
- 训练超参数的设置
- 评估和测试模型的方法

## 4. 使用工具
在这个部分，你会找到一些用于使用和部署大模型的工具和实用程序。这些工具可能包括：
- 模型推理脚本
- 文本生成示例
- 模型可视化工具
- 部署到生产环境的指南


