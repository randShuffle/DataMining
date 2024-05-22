# 基于大模型的实验

### 1.实验目标

给定二型糖尿病患者的EHR电子医疗记录，包含过往疾病记录(disease)、用药(medicine)、采取的措施(procedures)，预测该患者是否会在一年内发生心脏病或中风(CVD endpoint)。

### 2.数据集

使用Cradle数据集（私有数据集），一共有约12000个变量，均为0、1变量，代表患者曾经是否患有某种疾病、是否用过某种药以及是否采取过某种措施，label为0、1变量，表示患者是否会在一年内发生心脏病或中风。

### 3.Medical code的自然语言转换

有一份medical code对应于自然语言的映射表，比如说```snomed100000```表示患者曾经是否患有肥胖症，在输入给大模型进行零样本学习、少样本学习以及微调时，需要将medical code转换为自然语言。

### 4.微调大模型

#### 4.1实验步骤

1.在一台A40显卡的服务器(48G显存)上部署Llama2-7b开源大模型，其中Llama2大模型需要申请，随后在huggingface上下载。

![](C:\Users\Administrator\Desktop\exp\llama2.png)

2.找到Meta开源项目llama-recipes，在此基础上利用lora进行微调。

![](C:\Users\Administrator\Desktop\exp\llamarecipes.png)

Q：如何进行微调？

A：首先，先将训练集和测试集分成8：2，将medical code的feature全部转为自然语言，将其作为输入，患者是否会得心脏病作为输出。随后，执行下述命令行指令。整个训练过程大概持续3个小时。

```bash
python ../llama_finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```

核心代码如下：

```python

import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    print("1111111111111111111111111111111111111")
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
    print("22222222222222222222222222222222222222222")
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print("33333333333333333333333333333333333333333")
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    print("5555555555555555555555555555555555555555555555")
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    print("6666666666666666666666666666666666666666666")
    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)

```



3.对微调后大模型的输出进行解析。（二分类任务）

核心代码如下：

```python
import re
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize variables to store predictions and labels
predictions = []
labels = []

# Read the content of the result.txt file
with open('./train2000_epoch5_infer_200.txt', 'r',encoding='utf-8') as file:
    content = file.read()

# Split the content into individual tasks
tasks = content.split("### Instruction:")

# Define a function to extract the response 'Yes' or 'No' from the string
def extract_response(response_string):
    match = re.search(r'\b(Yes|No)\b', response_string)
    return match.group(1) if match else None

# Iterate through each task
# for task in tasks[1:]:
for task in tasks[1:]:
    # Split the task into lines
    lines = task.split('\n')

    # Find and extract response and label
    for i in range(len(lines)):
        if '### Response:' in lines[i]:
            response = extract_response(lines[i+1].strip())
        elif '### Label:' in lines[i]:
            label = lines[i+1].strip()
    
    # If the response is valid (either 'Yes' or 'No'), then append
    if response:
        predictions.append(response)
        labels.append(label)

print(len(predictions), len(labels))

# Calculate accuracy
total_samples = len(labels)
correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
accuracy = correct_predictions / total_samples

# Calculate precision, recall, and F1-score
precision = precision_score(labels, predictions, average='macro')
recall = recall_score(labels, predictions, average='macro')
f1 = f1_score(labels, predictions, average='macro')

# Calculate the confusion matrix
confusion = confusion_matrix(labels, predictions)

# Print the metrics
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print("Confusion Matrix:")
print(confusion)
```



输出如下：

![](C:\Users\Administrator\Desktop\exp\finetuneresult.png)

4.计算最终的评价指标。(acc,precision,recall,f1…)

首先先对得到的response进行正则表达式提取，Yes为1，No为0,随后计算accuracy、precision、recall以及f1-macro指标，混淆矩阵如下。

![](C:\Users\Administrator\Desktop\exp\finetunemetrics.png)

### 5. 采用In context learning的方法

采用gpt4作为基座大模型，在此基础上对比传统的机器学习方法、零样本学习、少样本学习以及双模型反馈的学习方法（后面会解释）。

机器学习方法采用决策树、逻辑回归、随机森林，分别在整个训练集上训练以及在6个训练集上训练（6个），在6个训练集上训练是为了和少样本学习（6个）进行比较。在整个训练集上训练是为了作为upper bound，与最后提出的模型反馈的学习方法进行比较。

双模型反馈的学习方法是指

1. **EHR-CoAgent框架**：所描述方法的名称，它使用多个语言学习模型（LLMs）的合作框架。这些模型共同工作以解决复杂问题，在本案例中是预测电子健康记录中的结果或模式。
2. **两部分系统**：该框架由两部分组成：
   - **预测代理PLLM**：这个代理负责做出初步预测。它可能会查看可用数据并试图预测未来的健康事件或患者病情的可能进展。
   - **评论代理KLLM**：这个代理充当观察者和预测代理输出的评审者。预测代理做出预测后，评论代理提供反馈，指出错误或确认正确的分析。
3. **反馈的整合**：系统的关键在于使用评论代理的反馈。这个反馈被纳入到预测代理的提示和过程中。通过这样做，预测代理应该能从错误中学习，有效地在上下文中进行训练。这个术语意味着代理是基于它接收的直接反馈进行调整，而不是通过一个单独的训练阶段。
4. **目标**：这个合作系统的最终目的是提高通过分析EHR所做出的疾病预测的准确性。其连续的预测和反馈循环旨在提升学习过程和优化系统的预测能力。

其中，不再展示零样本学习以及少样本学习的代码，下面展示一下coagent框架的核心代码：

```python
### critic_agent.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re
from evaluation_metrics import *
from parse_response import parse_response
import openai
import json
import time
import openai
import asyncio
from openai import AzureOpenAI
import time
import ast
from openai import AzureOpenAI


# def get_openai_api_key():
#     with open('../.openai_config', 'r') as config_file:
#         config = json.load(config_file)
#         return config['openai_api_key'], config['openai_api_endpoint']
    

def get_openai_api_key():
    with open('../.gpt35_config', 'r') as config_file:
        config = json.load(config_file)
        return config['openai_api_key'], config['openai_api_endpoint']

api_key, api_endpoint = get_openai_api_key()

client = AzureOpenAI(
    api_key = api_key,
    api_version = "2024-02-15-preview",
    azure_endpoint = api_endpoint
)

df = pd.read_csv('../dataset/cradle_cvd_deid_int.csv')
header = df.columns[1:]
with open('../dataset/cradle_cvd_deid_names.txt', 'r') as f:
    cradle_cvd_deid_names = f.readlines()
    cradle_cvd_deid_names = [x.strip() for x in cradle_cvd_deid_names]
cradle_code2name = dict(zip(header, cradle_cvd_deid_names))
shap_top_k = pd.read_csv('../result/singleFeature/shap_features.csv')['feature'].values.tolist()[:500]
shap_top_k.append('label')
df_part=df[shap_top_k]
df_part.drop_duplicates(inplace=True)
index_to_drop = df_part.index[(df_part == 0).all(axis=1)]
df_part.drop(index_to_drop,inplace=True)
print("cradle load.")
token_input=0
token_output=0

def info_generate(info:list)->str:
    snomed_regex = re.compile(r'^snomed.*', re.IGNORECASE)
    atc_regex = re.compile(r'^atc.*', re.IGNORECASE)
    cpt_regex = re.compile(r'^cpt.*', re.IGNORECASE)
    snomed_list=[]
    atc_list=[]
    cpt_list=[]

    # Loop through headers and identify column types using regex
    for medical_code in info:
        if snomed_regex.match(medical_code):
            snomed_list.append(medical_code)
        elif atc_regex.match(medical_code):
            atc_list.append(medical_code)
        elif cpt_regex.match(medical_code):
            cpt_list.append(medical_code)

    info_text="- Diagnoses made\n"
    if snomed_list==[]:
        info_text+='None\n'
    for i,code in enumerate(snomed_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'
   
    info_text+='- Medications prescribed\n'
    if atc_list==[]:
        info_text+='None\n'
    for i,code in enumerate(atc_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'

    info_text+='- Procedures performed\n'
    if cpt_list==[]:
        info_text+='None\n'
    for i,code in enumerate(cpt_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'

    return info_text    


def generate_1_code(row:pd.Series):
    code_1_list = row[row != 0].index.tolist()
    return code_1_list


def few_shots_generate(few_shots_label_0,few_shots_label_1):
    few_shots_ans=""
    case_num=1
   
    for index,shot in few_shots_label_0.iterrows():
        few_shots_ans+=f"### Case {case_num}\n"+info_generate(generate_1_code(shot))
        few_shots_ans+="- Prediction: No.\n"
        case_num+=1

    for index,shot in few_shots_label_1.iterrows():
        few_shots_ans+=f"### Case {case_num}\n"+info_generate(generate_1_code(shot))
        few_shots_ans+="- Prediction: Yes.\n"
        case_num+=1

    return few_shots_ans


def batch_generate(input_data_batch):
    batch_input_ans = ""
    case_index = 1
   
    for index, shot in input_data_batch.iterrows():
        patient_visit_data = ast.literal_eval(shot['visit_code'])
        label = shot['label']
        prediction = shot['response_text']

        batch_input_ans += f"### Case {case_index}\n" + "" + info_generate(patient_visit_data)
        if label == 1:
            batch_input_ans += "- Ground-Truth Result: Yes. The patient will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis of type 2 diabetes.\n"
        else:
            batch_input_ans += "- Ground-Truth Result: No. The patient will not develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis of type 2 diabetes.\n"
        batch_input_ans += f"- Prediction: {prediction}\n"

        batch_input_ans += f"\n"
        case_index += 1

    return batch_input_ans



def critic_agent(input_data_batch):
    message_text = [{"role": "system", "content": "You are an assistant who is good at self-reflection, gaining experience, and summarizing criteria. By reflecting on failure predictions that are given below, your task is to reflect on these incorrect predictions, compare them against the ground truth, and formulate criteria and guidelines to enhance the accuracy of future predictions."},
{"role": "user", "content":
f'''\
Task:
You will be given a batch of input data samples, where each sample is composed of three parts: the patient's medical history, the ground-truth result for whether the patient will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis of type 2 diabetes, and a wrong prediction. 
Please always remember that the predictions above are all incorrect. You should always use the ground truth as the final basis to discover many unreasonable aspects in the predictions and then summarize them into experience and criteria.
You are presented with the following: 
1. [Input Data] A batch of input data samples. Each data in the batch includes three parts: 
(a) the patient's medical history, including diseases that the patient has been diagnosed with, medications that the patient has taken, and procedures the patient has undergone.
(b) the ground-truth result for each patient's medical history on whether the patient will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis of type 2 diabetes.
(c) a wrong prediction. 

2. [Instructions] Instructions on suggesting criteria.

[Input Data]
{batch_generate(input_data_batch)}

[Instructions]
1. Please always remember that the predictions above are all incorrect. You should always use the ground truth as the final basis to discover many unreasonable aspects in the predictions and then summarize them into experience and criteria. For example, you may generate the following criteria, such as something that may not cause CVD or else.
2. Identify why the wrong predictions deviated from the ground truth by examining discrepancies in the medical history analysis.
3. Determine key and potential influencing factors, reasoning methods, and relevant feature combinations that could better align predictions with the ground truth.
4. The instructions should be listed in distinct rows, each representing a criteria or guideline.
5. The instructions should be generalizable to multiple samples, rather than specific to individual samples. 
6. Conduct detailed analysis and write criteria based on the input samples, rather than writing some criteria without foundation.
7. Please note that the criteria you wrote should not include the word "ground truth".
8. Generate the critera in such a format:
Something does not lead to a cardiovascular disease (CVD) endpoint within a year of the initial diagnosis.
Something leads to a cardiovascular disease (CVD) endpoint within a year of the initial diagnosis.

'''}
]
    
    while True:
        try:
            response = client.chat.completions.create(model="EGM-OpenAI-6-EHR", messages=message_text,temperature=0)
            #response = client.chat.completions.create(model="gpt-4-0125-Preview", messages=message_text,temperature=0,max_tokens=3000)
            break
        except openai.RateLimitError as e:
            print("Rate limit exceeded. Waiting for a while to retry...")
            time.sleep(60)
        except openai.BadRequestError as e:
            print(f"An error occurred while processing the image: {e}")
            response = "none"
            break

    return response


response_from_predict_agent = pd.read_csv('../result/test/gpt4_critic_agent_input.csv')
response_from_predict_agent=response_from_predict_agent[response_from_predict_agent['label']!=response_from_predict_agent['response_label']]
# Define the number of samples you want to take
N = 10 # Number of samples to take from the DataFrame
sampling_num = 10  # Number of times to sample from the DataFrame
criteria = []
for i in range(sampling_num):
    sampled_response = response_from_predict_agent.sample(n=N)
    response = critic_agent(sampled_response)
    criteria_batch = response.choices[0].message.content
    criteria.append(criteria_batch)
    token_input += response.usage.prompt_tokens
    token_output += response.usage.completion_tokens
    print(i)


tmp=pd.DataFrame()
tmp['criteria'] = criteria
tmp.to_csv('../result/test/gpt35_critic_agent.csv')
print(f'token_input:{token_input}')
print(f'token_output:{token_output}')

```

```python
### coagent.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re
from evaluation_metrics import *
from parse_response import *
import openai
from openai import AsyncAzureOpenAI
import asyncio
import json
import time
from summarize import *
import random


def get_openai_api_key():
    with open('../.openai_config', 'r') as config_file:
        config = json.load(config_file)
        return config['openai_api_key'], config['openai_api_endpoint']

api_key, api_endpoint = get_openai_api_key()

client = AsyncAzureOpenAI(
    api_key = api_key,
    api_version = "2024-02-15-preview",
    azure_endpoint = api_endpoint
)


# 目前测试集上取10%，小规模测试用
def data_loader(few_shot_0_path, few_shot_1_path, sampled_1_path, sampled_0_path):
    test_sample_n = 1000
    # test_sample_n = 10

    n_pos = int(test_sample_n * 0.2)  
    n_neg = test_sample_n - n_pos  

    df_few_shot_0 = pd.read_csv(few_shot_0_path,index_col=0).iloc[:2]
    df_few_shot_1 = pd.read_csv(few_shot_1_path,index_col=0).iloc[:2]
    df_test_pos = pd.read_csv(sampled_1_path,index_col=0).sample(n=n_pos, random_state=1)
    df_test_neg = pd.read_csv(sampled_0_path,index_col=0).sample(n=n_neg, random_state=1)
    df_test_all = pd.concat([df_test_pos, df_test_neg])

    return df_few_shot_0, df_few_shot_1, df_test_all
    

def info_generate(info:list)->str:
    snomed_regex = re.compile(r'^snomed.*', re.IGNORECASE)
    atc_regex = re.compile(r'^atc.*', re.IGNORECASE)
    cpt_regex = re.compile(r'^cpt.*', re.IGNORECASE)
    snomed_list=[]
    atc_list=[]
    cpt_list=[]

    # Loop through headers and identify column types using regex
    for medical_code in info:
        if snomed_regex.match(medical_code):
            snomed_list.append(medical_code)
        elif atc_regex.match(medical_code):
            atc_list.append(medical_code)
        elif cpt_regex.match(medical_code):
            cpt_list.append(medical_code)

    info_text="- Diagnoses made\n"
    if snomed_list==[]:
        info_text+='None\n'
    for i,code in enumerate(snomed_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'
   
    info_text+='- Medications prescribed\n'
    if atc_list==[]:
        info_text+='None\n'
    for i,code in enumerate(atc_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'

    info_text+='- Procedures performed\n'
    if cpt_list==[]:
        info_text+='None\n'
    for i,code in enumerate(cpt_list):
        info_text+=str(i+1)+'. '+cradle_code2name[code]+'\n'

    return info_text    


def generate_1_code(row:pd.Series):
    code_1_list = row[row != 0].index.tolist()
    return code_1_list


def few_shots_generate(few_shots_label_0,few_shots_label_1):
    few_shots_ans=""
    case_num=1
   
    for index,shot in few_shots_label_0.iterrows():
        few_shots_ans+=f"### Case {case_num}\n"+info_generate(generate_1_code(shot))
        few_shots_ans+="- Prediction: No.\n"
        case_num+=1

    for index,shot in few_shots_label_1.iterrows():
        few_shots_ans+=f"### Case {case_num}\n"+info_generate(generate_1_code(shot))
        few_shots_ans+="- Prediction: Yes.\n"
        case_num+=1

    return few_shots_ans


async def enhanced_predict_agent(code_1_list, few_shots_label_0, few_shots_label_1):
   
    message_text = [
{"role": "system", "content": "You are a medical expert with a specialization in type 2 diabetes and cardiovascular disease. Your task is to predict whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis."},
{"role": "user", "content":
f'''\
Task:
Your task is to predict whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis based on the provided patient's medical history. You will be presented with a patient's medical history and various resources to aid in your prediction. Please provide your reasoning and make your prediction by learning from the resources. 
You are presented with the following: 
1. [Attention] A cautionary note. When evaluating whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis, you should comprehensively consider the interaction of variable factors, including not only diseases, but also medicines, or procedures the patient has undergone. You should avoid premature conclusion due to excessive caution or conservative reasoning. When predicting, you should avoid pessimism and maintain an objective attitude. Objective analysis can help you make more accurate predictions.
2. [Criteria and guidelines to boost prediction performance] These criteria and guidelines are generated by observing and accumulating successful prediction experience and reflecting on failure predictions ways for improving prediction performance. When you predict and reason, you should consider these ways to help you think more comprehensively.
3. [CVD Endpoint Definition] The definition of the prediction target: cardiovascular disease (CVD) endpoint. 
4. [Past Medical History] Patient's past medical history, which captures specific diagnoses made, medications prescribed, and procedures performed. 
5. [Instructions] Guidelines on how to analyze the patient's medical history, provide reasoning, and make predictions. This includes referring to the attention note, demonstration cases, and exploring the interaction of various factors and the interplay between diseases, medications, and procedures that the patient has undergone. The reasoning process should support and aid in the final prediction for a CVD endpoint.
6. [Demonstration Cases] Some real and typical cases, including the patient's medical history (diseases, medications, and procedures) and the ground truth result of whether the patients with type 2 diabetes experience cardiovascular disease (CVD) endpoint within a year after the initial diagnosis. 
7. [Output Format] The required format for your response. Please ensure that you strictly adhere to the format requirements. You must provide a confirmed prediction by choosing between "Yes" or "No".


[Attention]
You should avoid always taking a cautious and pessimistic attitude, which will lead to one-sided predictions. In addition, your inference should not be too cautious and too biased towards safety. Therefore, you should not continue to take the position of conservative prediction, which may deprive you of other opportunities for insight. Instead, objective analysis can help you make more accurate predictions. When evaluating whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis, you should comprehensively consider the interaction of variable factors, including not only diseases, but also medicines, or procedures the patient has undergone.


[Criteria and guidelines to boost prediction performance]
xxx

[CVD Endpoint Definition]
A CVD endpoint is identified by the presence of coronary heart disease (CHD), congestive heart failure (CHF), myocardial infarction (MI), or stroke.


[Past Medical History]
{info_generate(code_1_list)}


[Instructions]
Based on the patient's past medical history, provide reasoning on whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis. 
Please: 
(1) use your knowledge
(2) follow the attention note
(3) refer to the criteria and guidelines to boost prediction performance
(4) learn from the provided demonstration cases
(5) To predict whether a patient with type 2 diabetes will develop a cardiovascular disease (CVD) endpoint within a year of their initial diagnosis, a comprehensive analysis of the patient's medical history is required. While considering the interplay of diseases, medications, and procedures, make sure to:
Acknowledge any aspects of the patient's history that mitigate the risk of developing a CVD endpoint within the stipulated timeframe.
Identify and weigh both risk factors and protective factors evident in the patient's medical history.
Consider the presence of any comorbid conditions that may independently increase or decrease the risk of CVD.
Examine the patient's medication profile to discern any pharmacological interventions that may alter the course of disease progression.
Evaluate any medical or surgical procedures the patient has undergone that could impact their cardiovascular health.
Utilize a multihop and step-by-step reasoning approach to systematically analyze the data.


[Demonstration Cases]
{few_shots_generate(few_shots_label_0,few_shots_label_1)}


[Output Format]
Your final response should include:

1. Prediction: Conclude your analysis with a clear and concise prediction. This prediction must be a single word, either "Yes" or "No", indicating whether you believe the patient is likely to develop a CVD endpoint within a year of their initial diabetes diagnosis. This prediction should be the first line of your response, to facilitate easy parsing.

2. Reasoning: Provide a detailed reasoning process.

Ensure that your analysis is thorough and based on the information provided, leading logically to your final prediction.

'''}
    ]


    while True:
        try:
            
            # response = await client.chat.completions.create(model="EGM-OpenAI-6-EHR", messages=message_text, temperature=0, logprobs=True, top_logprobs=5)
            response = await client.chat.completions.create(model="gpt-4-0125-Preview", messages=message_text, temperature=0, logprobs=True, top_logprobs=5)
            # If the request is successful, break the loop
            break
        except openai.RateLimitError as e:
            print("Rate limit exceeded. Waiting for a while to retry...")
            time.sleep(random.randint(60, 120))
        except:
            print("other exceptions")
            time.sleep(random.randint(60, 120))

    return response


async def call_client(df_test_all, predict_agent, generate_1_code, df_few_shot_0, df_few_shot_1):
    batch_size = 10
    test_sample_num = df_test_all.shape[0]
    response_text_list, response_label_list, positive_proba_list = [], [], []
    token_input, token_output = 0, 0

    for i in range(0, test_sample_num // batch_size):
        batch = df_test_all.iloc[i * batch_size:(i + 1) * batch_size]
        tasks = [predict_agent(generate_1_code(tc), df_few_shot_0, df_few_shot_1) for index, tc in batch.iterrows()]
        responses = await asyncio.gather(*tasks)
        print(i)
        for response in responses:
            message, response_label, pos_prob, token_nums = parse_response(response)
            response_text_list.append(message)
            if response_label is not None:
                response_label_list.append(response_label)
                positive_proba_list.append(pos_prob)
            else:
                handle_user_input(response_text_list, response_label_list, positive_proba_list)
            token_input += token_nums[0]
            token_output += token_nums[1]

    return response_text_list, response_label_list, positive_proba_list



df = pd.read_csv('../dataset/cradle_cvd_deid_int.csv')
header = df.columns[1:]
with open('../dataset/cradle_cvd_deid_names.txt', 'r') as f:
    cradle_cvd_deid_names = f.readlines()
    cradle_cvd_deid_names = [x.strip() for x in cradle_cvd_deid_names]
cradle_code2name = dict(zip(header, cradle_cvd_deid_names))

shap_top_k = pd.read_csv('../result/singleFeature/shap_features.csv')['feature'].values.tolist()[:500]
shap_top_k.append('label')
df_part = df[shap_top_k]
df_part.drop_duplicates(inplace=True)

index_to_drop = df_part.index[(df_part == 0).all(axis=1)]
df_part.drop(index_to_drop, inplace=True)
print("cradle load.")

df_few_shot_0, df_few_shot_1, df_test_all = data_loader('../dataset/fexemplar_label_0.csv', '../dataset/fexemplar_label_1.csv', '../dataset/sampled_label_1.csv', '../dataset/sampled_label_0.csv')

# input
tmp = pd.DataFrame()
visit_code_list = []
visit_info_list = []
for index, tc in df_test_all.iterrows():
    visit_code_list.append(generate_1_code(tc))
    visit_info_list.append(info_generate(generate_1_code(tc)))
tmp['visit_code'] = visit_code_list
tmp['visit_info'] = visit_info_list
tmp['label'] = df_test_all['label'].values

# predict output
response_text_list, response_label_list, positive_proba_list = asyncio.run(call_client(df_test_all, enhanced_predict_agent, generate_1_code, df_few_shot_0, df_few_shot_1))
tmp['response_text'] = response_text_list
tmp['response_label'] = response_label_list
tmp['positive_proba'] = positive_proba_list
output_file_path = '../result/test/enhanced_predictor_agent.csv'
tmp.to_csv(output_file_path, index=False)

# evaluation
df_predictor_agent = pd.read_csv('../result/test/enhanced_predictor_agent.csv')
accuracy, precision, recall, f1, balanced_accuracy, auroc, aupr = metrics(df_predictor_agent['label'].values, df_predictor_agent['response_label'].values, df_predictor_agent['positive_proba'].values)
print(f"accuracy: {accuracy * 100:.2f}%, precision: {precision * 100:.2f}%, recall: {recall * 100:.2f}%, f1: {f1 * 100:.2f}%, balanced_accuracy: {balanced_accuracy * 100:.2f}%, auroc: {auroc * 100:.2f}%, aupr: {aupr * 100:.2f}%")

```



EHR-CoAgent的框架使用了两个LLM代理：一个是进行预测并生成推理过程的预测器代理，另一个是分析错误预测并提供改进指导的批评家代理。评论家代理的反馈用于更新给预测代理的提示，使系统能够从错误中吸取教训，并适应基于EHR的疾病预测任务的特定挑战。具体演示如下：

![](C:\Users\Administrator\Desktop\exp\coagent.png)

选取一个prompt如下：

![](C:\Users\Administrator\Desktop\exp\coagentprompt.png)

批评家总结出的规则展示以下几条：

![](C:\Users\Administrator\Desktop\exp\criteria.png)

最终实验的指标如下：

![](C:\Users\Administrator\Desktop\exp\finalmetrics.png)

### 6.实验结论

我们研究了大型语言模型（LLM）在基于电子健康记录（EHR）的疾病预测任务中的应用。我们使用lora微调、各种提示策略评估了LLM的零样本和少样本诊断性能，并提出了一种结合预测代理和批评代理的新型协作方法。这种方法使系统能够从错误中吸取教训，并适应基于EHR的疾病预测的挑战。我们的工作强调了LLM作为临床决策支持工具的潜力，并有助于开发高效的疾病预测系统，该系统可以在最少的训练数据下运行。