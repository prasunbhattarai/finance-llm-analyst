import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
from peft import PeftModel


def load_finetuned_llm(cfg):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_model_dir = os.path.join(project_root, cfg["model"]["cache_dir"])  # "../models/fined_tuned" from config
    adapter_path  = os.path.join(project_root, cfg["output"]["dir"])  # "../models/fined_tuned" from config
    model_name= "Qwen/Qwen2.5-3B-Instruct"
    tokenizer= AutoTokenizer.from_pretrained(cfg["output"]["dir"])

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir = base_model_dir,
        device_map= "auto",
        torch_dtype= torch.float16
    )

    model = PeftModel.from_pretrained(base_model, adapter_path )    

    pipe = pipeline(
        "text-generation",
        model= model,
        tokenizer= tokenizer,
        max_new_tokens= 256,
        temperature= 0.2,
        do_sample= False
    )
    return HuggingFacePipeline(pipeline=pipe)