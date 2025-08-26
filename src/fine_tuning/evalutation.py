import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, pipeline
from peft import PeftModel
import yaml
from datasets import load_from_disk
import math

def load_config(path: str):
    with open (path, 'r') as cfgs:
        return yaml.safe_load(cfgs)
    
def load_base_model(cfg):
    load_cfg= cfg["model"]
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
    ).eval()
    return model


def load_merged_finetuned_model(cfg):
    temp_model = load_base_model(cfg)
    adapter_path = cfg["output"]["dir"]
    fined_tuned = PeftModel.from_pretrained(temp_model, adapter_path)
    fined_tuned = fined_tuned.merge_and_unload().eval()
    return fined_tuned

def build_eval_loader(cfg):
    dataset= load_from_disk(cfg["data"]["test_data"])
    tokenized_test = dataset.with_format('torch')
    eval_loader = DataLoader(
        tokenized_test,
        batch_size= 8,
        collate_fn= default_data_collator
    )
    return eval_loader

import math
@torch.no_grad()
def compute_perplexity(model, eval_loader, pad_token_id):
    losses = []
    for batch in eval_loader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        if "labels" in batch:
            batch["labels"][batch["labels"] == pad_token_id] = -100


        loss = model(**batch).loss
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))

def run_eval(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    base_model = load_base_model(cfg)
    fine_tuned = load_merged_finetuned_model(cfg)
    eval_loader = build_eval_loader(cfg)

    base_ppl = compute_perplexity(base_model, eval_loader, tokenizer.pad_token_id)
    tuned_ppl = compute_perplexity(fine_tuned, eval_loader, tokenizer.pad_token_id)

    print(f"Base Model Perplexity: {base_ppl:.4f}")
    print(f"Tuned Model Perplexity: {tuned_ppl:.4f}")


if __name__ == "__main__":
    cfg = load_config("../configs/qwen.yaml")
    run_eval(cfg)