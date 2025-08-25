import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig



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

def load_tokenier(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_lora_config(cfg):
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r = lora_cfg ["r"],
        lora_alpha= lora_cfg ["lora_alpha"],
        target_modules= lora_cfg ["target_modules"],
        lora_dropout= lora_cfg ["lora_dropout"],
        bias= lora_cfg["bias"],
        task_type= lora_cfg["task_type"]
    )
    return lora_config


def get_training_config(cfg):
    load_cfg = cfg["training"]
    training_config = SFTConfig(
        num_train_epochs= load_cfg["num_train_epochs"],
        per_device_train_batch_size= load_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size= load_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps= load_cfg["gradient_accumulation_steps"],
        learning_rate= load_cfg["learning_rate"],
        fp16= load_cfg["fp16"],
        eval_steps= load_cfg["eval_steps"],
        logging_steps= load_cfg["logging_steps"],
        save_strategy= "epoch",
        remove_unused_columns= load_cfg["remove_unused_columns"],
        label_names= load_cfg["label_names"]
    )
    return training_config

def train(cfg):
    dataset= load_from_disk(cfg["data"]["path"])
    model= load_model(cfg)
    tokenizer= 0 # TODO: Add tokenizer
    lora_config = load_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    training_config = get_training_config(cfg)

    trainer = SFTTrainer(
        model= model,
        args= training_config,
        tokenizer= tokenizer,
        train_dataset= dataset["train"],
        eval_dataset= dataset["test"],
    )

    trainer.train()
    model.save_pretrained(cfg["output"]["dir"])
    tokenizer.save_pretrained(cfg["output"]["dir"])


if __name__ == "__main__":
    cfg = load_config("./configs/qwen.yaml")
    train(cfg)