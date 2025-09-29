import os
import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_config(filename: str) -> dict:
    """
    Load a YAML configuration file from the project's configs directory.

    Args:
        filename (str): Name of the YAML config file (e.g., "qwen.yaml").

    Returns:
        dict: Parsed configuration as a Python dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    config_path = os.path.join(project_root, "configs", filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as cfgs:
        return yaml.safe_load(cfgs)


def load_model(cfg: dict):
    """
    Load a pretrained causal language model with quantization settings.

    Args:
        cfg (dict): The configuration dictionary containing model parameters.

    Returns:
        AutoModelForCausalLM: The loaded model ready for training or inference.
    """
    load_cfg = cfg["model"]
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=getattr(
            torch, load_cfg["quantization"]["bnb_4bit_compute_dtype"]
        ),
        bnb_4bit_quant_type=load_cfg["quantization"]["bnb_4bit_quant_type"],
        load_in_4bit=load_cfg["quantization"]["load_in_4bit"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_cfg["model_name"],
        cache_dir=load_cfg["cache_dir"],
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    return model


def load_lora_config(cfg: dict) -> LoraConfig:
    """
    Create a LoRA configuration object for parameter-efficient fine-tuning.

    Args:
        cfg (dict): The configuration dictionary containing LoRA parameters.

    Returns:
        LoraConfig: A LoRA configuration object for PEFT.
    """
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config


def get_training_config(cfg: dict) -> TrainingArguments:
    """
    Build Hugging Face TrainingArguments from the configuration.

    Args:
        cfg (dict): The configuration dictionary containing training parameters.

    Returns:
        TrainingArguments: A configured TrainingArguments object.
    """
    load_cfg = cfg["training"]
    training_config = TrainingArguments(
        num_train_epochs=load_cfg["num_train_epochs"],
        per_device_train_batch_size=load_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=load_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=load_cfg["gradient_accumulation_steps"],
        learning_rate=float(load_cfg["learning_rate"]),
        fp16=load_cfg["fp16"],
        eval_strategy=load_cfg["eval_strategy"],
        eval_steps=load_cfg["eval_steps"],
        logging_steps=load_cfg["logging_steps"],
        save_strategy=load_cfg["save_strategy"],
        label_names=load_cfg["label_names"],
        logging_dir=load_cfg["logging_dir"],
        remove_unused_columns=load_cfg["remove_unused_columns"],
    )
    return training_config


def train(cfg: dict):
    """
    Train a quantized language model with LoRA fine-tuning using the Hugging Face Trainer API.

    Steps:
        1. Load dataset from disk.
        2. Load base model with quantization.
        3. Apply LoRA parameter-efficient fine-tuning.
        4. Configure training arguments.
        5. Train with a train/eval dataset split.
        6. Save the trained model to output directory.

    Args:
        cfg (dict): Configuration dictionary containing data, model,
                    LoRA, and training settings.
    """
    dataset = load_from_disk(cfg["data"]["train_data"])
    model = load_model(cfg)
    lora_config = load_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    training_config = get_training_config(cfg)

    # Small train/eval split for demonstration
    train_dataset = dataset.shuffle(seed=42).select(range(2000))
    eval_dataset = dataset.shuffle(seed=42).select(range(2000, 2200))

    trainer = Trainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(cfg["output"]["dir"])


if __name__ == "__main__":
    """
    Entry point for running training using the specified YAML config.
    """
    cfg = load_config("qwen.yaml")
    train(cfg)
