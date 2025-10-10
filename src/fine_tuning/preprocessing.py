from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import yaml, os

def load_config(filename: str):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "configs", filename)
    with open(config_path, "r") as cfgs:
        return yaml.safe_load(cfgs)
    
cfg = load_config("../configs/qwen.yaml")

def finance_format(dataset):
    rules = dataset["rules"]   
    instruction = dataset["question"]
    response = dataset["answer"]
    prompt = f"### \n\n### Question:\n{instruction}"
    return {"text": f"{prompt}\n\n### Response:\n{response}"}

def dolly_format(dataset):
    instruction = dataset["instruction"]
    context = dataset.get("context", "")
    response = dataset["response"]
    if context:
        prompt = f"Instruction: {instruction}\nContext: {context}"
    else:
        prompt = f"Instruction: {instruction}"
    return {"text": f"### {prompt}\n\n### Response:\n{response}"}

def fiqa_format(dataset):
    instruction = dataset["instruction"]
    inp = dataset["input"]
    output = dataset["output"]
    prompt = f"Instruction: {instruction}\nInput: {inp}"
    return {"text": f"### {prompt}\n\n### Response:\n{output}"}

def tokenize(batch, tokenizer):
    token = tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    token["labels"] = token["input_ids"].copy()
    return token

def prepare_datasets(cfg: dict, test_size: float = 0.1, seed: int = 42, target_size = 1700):
    dolly_dataset = load_dataset("databricks/databricks-dolly-15k", cache_dir=cfg["data"]["dolly_dir"])["train"]
    fiqa_dataset = load_dataset("llamafactory/fiqa", cache_dir=cfg["data"]["fiqa_dir"])["train"]
    finance_dataset = load_dataset("json", data_files={"train": cfg["data"]["synthetic_dir"]})["train"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    dolly_dataset = dolly_dataset.shuffle(seed=seed).select(range(target_size))
    fiqa_dataset = fiqa_dataset.shuffle(seed=seed).select(range(target_size))
    finance_dataset = finance_dataset.shuffle(seed=seed).select(range(target_size))

    dolly_dataset = dolly_dataset.map(dolly_format, remove_columns=dolly_dataset.column_names)
    fiqa_dataset = fiqa_dataset.map(fiqa_format, remove_columns=fiqa_dataset.column_names)
    finance_dataset = finance_dataset.map(finance_format, remove_columns=finance_dataset.column_names)

    dolly_split = dolly_dataset.train_test_split(test_size=test_size, seed=seed)
    fiqa_split = fiqa_dataset.train_test_split(test_size=test_size, seed=seed)
    finance_split = finance_dataset.train_test_split(test_size=test_size, seed=seed)

    tokenized_dolly_train = dolly_split["train"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    tokenized_dolly_test = dolly_split["test"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])

    tokenized_fiqa_train = fiqa_split["train"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    tokenized_fiqa_test = fiqa_split["test"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])

    tokenized_finance_train = finance_split["train"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    tokenized_finance_test = finance_split["test"].map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["text"])

    train_dataset = concatenate_datasets([tokenized_dolly_train, tokenized_fiqa_train, tokenized_finance_train]).shuffle(seed=seed)
    test_dataset = concatenate_datasets([tokenized_dolly_test, tokenized_fiqa_test, tokenized_finance_test]).shuffle(seed=seed)

    train_dataset.save_to_disk(cfg["data"]["train_data"])
    test_dataset.save_to_disk(cfg["data"]["test_data"])

    tokenizer.save_pretrained(cfg["output"]["dir"])


    return train_dataset, test_dataset, tokenizer


if __name__ == "__main__":
    train_dataset, test_dataset, tokenizer = prepare_datasets(cfg)
