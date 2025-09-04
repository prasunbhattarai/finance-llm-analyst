import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


def load_finetuned_llm(cfg):
    tokenizer= AutoTokenizer.from_pretrained(cfg["model"]["model_name"])

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_name"],
        device_map= "auto",
         torch_dtype= torch.float16
    )

    pipe = pipeline(
        "text-generation",
        model= base_model,
        tokenizer= tokenizer,
        max_new_tokens= 256,
        temperature= 0.2,
        do_sample= False
    )
    return HuggingFacePipeline(pipeline=pipe)