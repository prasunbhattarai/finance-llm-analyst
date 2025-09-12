import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import os
from peft import PeftModel

def tokenizer(cfg):
    model_name = cfg["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir ='./tokens', trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

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
    # fined_tuned = fined_tuned.merge_and_unload().eval()
    return fined_tuned.eval()



def loaded_model(cfg):
    pipe = pipeline(
        "text-generation",
        model = load_merged_finetuned_model(cfg),
        tokenizer=tokenizer(cfg),
        max_new_tokens=256,   
        temperature=0.7,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)



