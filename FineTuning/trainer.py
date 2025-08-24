import torch
import yaml 
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType



def load_config(path: str):
    with open (path, 'r') as cfgs:
        return yaml.safe_load(cfgs)
    
def load_model(cfg):
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype = getattr(torch, cfg["model"]["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type = cfg["model"]["quantization"]["bnb_4bit_quant_type"],
        load_in_4bit = cfg["model"]["quantization"]["load_in_4bit"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name"],
        cache_dir = cfg["model"]["cache_dir"],
        quantization_config = bnb_config,
        trust_remote_code = True,
        device_map = "auto"
    )
    return model

# TODO: Tokenization


if __name__ == "__main__":
    cfg = load_config("./configs/qwen.yaml")
    load_model(cfg)