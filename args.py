from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    """
        Model arguments
    """
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        },
    )


@dataclass
class DataArguments:
    """
        Data arguments
    """
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "The path of the dataset to use."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
        Training arguments
    """
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=8)
    lora_target: Optional[str] = field(
        default="all",
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the linear modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    lora_alpha: Optional[int] = field(
        default=0
    )
    lora_dropout: Optional[float] = field(default=0.0)
    

    