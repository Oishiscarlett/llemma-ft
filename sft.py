#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Shan Haojia

import transformers
import torch
from args import DataArguments, TrainingArguments, ModelArguments
from typing import Dict, Sequence
from torch.utils.data import Dataset
import logging
import json
import random
from dataclasses import dataclass
import copy
from transformers import Trainer
from peft import LoraConfig, TaskType, get_peft_model
from callbacks import LogCallback


# 常见的填充词
IGNORE_TOKENS = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
}


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    主要是从文件中取出数据，并为其加上prompt
    """
    
    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()    
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            list_data_dict = json.load(open(data_path, 'r'))
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]
        
        list_data_dict = random.sample(list_data_dict, len(list_data_dict))

        # 初步处理数据
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input", "prompt_no_input"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]

        # 为每一个output加上结束符
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        
        self.sources = sources
        self.targets = targets
    
    def __len__(self):
        return len(self.sources)
    
    def __naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __naive__call(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 创建了两个列表，一个列表包含了instances中所有的input_ids，另一个包含所有的label
        input_ids, labels = tuple([instances[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # 处理输入和标签：对于输入，使用pad将所有输入扩充到相同长度
        # 对于labels，使用ignore遮盖
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pack_sequence(labels, batch_first=True, padding_value=IGNORE_TOKENS)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id) #这里返回一个张量，其中1的位置表示非填充，0表示填充
        )
        

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            sources.append(instance['input_ids'])
            targets.append(instance['labels'])
        
        # 对数据预处理
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict["input_ids"], data_dict["labels"]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_TOKENS
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id) #1是内容，0是pad
        )
        
        


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    Adopted from https://github.com/zhangir-azerbayev/MetaMath/blob/62916b8ae2ea0057f2d440b23d4e29cb5971b17b/train_math.py
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # 获取输入输出的嵌入权重
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算现有嵌入的平均值
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        
        # 为新令牌设置嵌入值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """preprocess the data by tokenizing"""
    examples = [s + t for s, t in zip(sources, targets)]
    # tokenize
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, sources_len in zip(labels, sources_tokenized["input_ids"]):
        label[:sources_len] = IGNORE_TOKENS
    return dict(input_ids=input_ids, labels=labels)
    
    
    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt", # 返回pytorch张量；如果是tensorflow，输入tf
            padding="longest", # 将所有输入填充到和最长字符一样长
            max_length=tokenizer.model_max_length,
            truncation=True # 保证模型输入不会太长，会截断
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
      tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens 
    )
    

def make_supervised_dataset_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    # 首先要写一个解析器，用于解析输入的参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载model
    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )

    # 如果tokenizer中没有pad_token，则添加pad_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model
        )

    # 给tokenizer添加上特殊token
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN
        }
    )
    
    # 处理数据，获取数据
    data_module = make_supervised_dataset_module(tokenizer, data_args)
    
    
    # 使用lora训练
    if training_args.use_lora:
        peft_kwargs = {
            "r": training_args.lora_rank,
            "target_modules": training_args.lora_target,
            "lora_alpha": training_args.lora_alpha,
            "lora_dropout": training_args.lora_dropout
        }
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_kwargs
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    
    # 设置log
    callbacks = [LogCallback()]
    
    # 训练
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks
        **data_module
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    
    

if __name__ == "__main__":
    train()
