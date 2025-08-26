import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer,TrainingArguments,Trainer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType



def load_config(path: str):
    with open (path, 'r') as cfgs:
        return yaml.safe_load(cfgs)
    
def load_model(cfg):
    load_cfg = cfg["model"]
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype = getattr(torch, load_cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type = load_cfg["quantization"]["bnb_4bit_quant_type"],
        load_in_4bit = load_cfg["quantization"]["load_in_4bit"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_cfg["model_name"],
        cache_dir = load_cfg["cache_dir"],
        quantization_config = bnb_config,
        trust_remote_code = True,
        device_map = "auto"
    )
    return model

def load_lora_config(cfg):
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r = lora_cfg ["r"],
        lora_alpha= lora_cfg ["lora_alpha"],
        target_modules= lora_cfg ["target_modules"],
        lora_dropout= lora_cfg ["lora_dropout"],
        bias= lora_cfg["bias"],
        task_type= TaskType.CAUSAL_LM
    )
    return lora_config


def get_training_config(cfg):
    load_cfg = cfg["training"]
    training_config = TrainingArguments(
        num_train_epochs= load_cfg["num_train_epochs"],
        per_device_train_batch_size= load_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size= load_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps= load_cfg["gradient_accumulation_steps"],
        learning_rate= float(load_cfg["learning_rate"]),
        fp16= load_cfg["fp16"],
        eval_strategy = load_cfg["eval_strategy"],
        eval_steps= load_cfg["eval_steps"],
        logging_steps= load_cfg["logging_steps"],
        save_strategy= load_cfg["save_strategy"],
        label_names = load_cfg["label_names"],
        remove_unused_columns= load_cfg["remove_unused_columns"],

    )
    return training_config

def train(cfg):
    dataset= load_from_disk(cfg["data"]["train_data"])
    model= load_model(cfg)
    lora_config = load_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    # Load training config
    training_config = get_training_config(cfg)
 
    train_dataset = dataset.shuffle(seed=42).select(range(2000))
    eval_dataset = dataset.shuffle(seed=42).select(range(2000, 2200))

    trainer = Trainer(
        model= model,
        args= training_config,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset   

    )

    trainer.train()
    model.save_pretrained(cfg["output"]["dir"])



if __name__ == "__main__":
    cfg = load_config("../configs/qwen.yaml")
    train(cfg)