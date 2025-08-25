from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml

def load_config(path: str):
    with open (path, 'r') as cfgs:
        return yaml.safe_load(cfgs)
    
cfg = load_config("./configs/qwen.yaml")

dolly_dataset = load_dataset("databricks/databricks-dolly-15k", cache_dir="./datasets/raw/dolly")
fiqa_dataset = load_dataset("llamafactory/fiqa", cache_dir="./datasets/raw/fiqa")

tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
tokenizer.pad_token = tokenizer.eos_token

def dolly_format(dataset):
    instruction = dataset["instruction"]
    context = dataset.get("context", "")
    response = dataset["response"]

    if context:
        prompt = f"Instruction:{instruction}\nContext:{context}\nAnswer:"
    else:
        prompt = f"Instruction:{instruction}\nAnswer:"

    return {
        "text": f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    }


def fiqa_format(dataset):
    instruction = dataset["instruction"]
    inp= dataset["input"]
    output= dataset["output"]
    prompt = f"Instruction:{instruction}\nInput:{inp}\nAnswer:"

    
    return{
        "text": f"### Instruction:\n{prompt}\n\n### Response:\n{output}"
    }


def tokenize(batch):
    token = tokenizer(
        batch["text"],
        truncation = True,
        max_length = 256,
        padding = "max_length",
    )
    token["labels"] = token["input_ids"].copy()
    return token


dolly_dataset = dolly_dataset["train"].map(dolly_format, remove_columns=dolly_dataset["train"].column_names)
fiqa_dataset = fiqa_dataset["train"].map(fiqa_format, remove_columns=fiqa_dataset["train"].column_names)

tokenized_dolly = dolly_dataset.map(tokenize, batched=True, remove_columns=["text"]).select(range(1100))
tokenized_fiqa = fiqa_dataset.map(tokenize, batched=True, remove_columns=["text"]).select(range(1100))

tokenized_dataset= concatenate_datasets([tokenized_dolly,tokenized_fiqa])

tokenized_dataset.save_to_disk("./datasets/processed")

tokenizer.save_pretrained(cfg["output"]["dir"])

