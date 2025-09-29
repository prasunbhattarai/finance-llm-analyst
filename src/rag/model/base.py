import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
import os
from peft import PeftModel


def tokenizer(cfg):
    """
    Load and configure a tokenizer for the specified model.

    - Loads the tokenizer from Hugging Face Hub (or cache).
    - Sets the pad token to EOS to ensure compatibility with generation.

    Args:
        cfg (dict): Configuration dictionary containing model details.

    Returns:
        AutoTokenizer: Hugging Face tokenizer instance.
    """
    model_name = cfg["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir='./tokens', trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg):
    """
    Load a quantized base causal language model with 4-bit precision.

    Uses BitsAndBytesConfig for memory-efficient loading and places the
    model automatically on available devices.

    Args:
        cfg (dict): Configuration dictionary with model and quantization details.

    Returns:
        AutoModelForCausalLM: The base causal language model in evaluation mode.
    """
    load_cfg = cfg["model"]
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=getattr(torch, load_cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=load_cfg["quantization"]["bnb_4bit_quant_type"],
        load_in_4bit=load_cfg["quantization"]["load_in_4bit"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_cfg["model_name"],
        cache_dir=load_cfg["cache_dir"],
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    return model


def load_merged_finetuned_model(cfg):
    """
    Load the base model and merge it with fine-tuned LoRA adapters.

    Args:
        cfg (dict): Configuration dictionary containing output model details.

    Returns:
        PeftModel: Fine-tuned model with adapters loaded, set to eval mode.
    """
    temp_model = load_base_model(cfg)
    adapter_source = cfg["output"]["hf_repo"]
    fined_tuned = PeftModel.from_pretrained(
        temp_model, adapter_source, cache_dir=cfg["output"]["dir"]
    )
    # fined_tuned = fined_tuned.merge_and_unload().eval()  # Optional merge
    return fined_tuned.eval()


def loaded_model(cfg):
    """
    Build a HuggingFacePipeline for text generation with the fine-tuned model.
    Steps:
    - Loads the merged fine-tuned model.
    - Wraps it in a text-generation pipeline.
    - Applies generation settings (max tokens, temperature, deterministic).

    Args:
        cfg (dict): Configuration dictionary with model and tokenizer details.

    Returns:
        HuggingFacePipeline: A LangChain-compatible pipeline for inference.
    """
    pipe = pipeline(
        "text-generation",
        model=load_merged_finetuned_model(cfg),
        tokenizer=tokenizer(cfg),
        max_new_tokens=256,
        temperature=0.7,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)
