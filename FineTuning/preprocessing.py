from datasets import load_dataset
dataset = load_dataset("databricks/databricks-dolly-15k", cache_dir="./datasets/raw")
def format(dataset):
    instruction = dataset["instruction"]
    context = dataset.get("context", "")
    response = dataset["response"]

    if context:
        prompt = f"Instruction:{instruction}\nContext:{context}\nAnswer:"
    else:
        prompt = f"Intruction:{instruction}\nAnswer:"

    return {"prompt": prompt, "completion": response }

dataset = dataset["train"].map(format)

dataset.save_to_disk("./datasets/processed")

