import os
import yaml
from src.rag.pipeline import main

def load_config(filename: str):
    """
    Load a YAML config file from the project's configs directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "configs", filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as cfgs:
        return yaml.safe_load(cfgs)

if __name__ == "__main__":
    main(load_config("qwen.yaml"))
