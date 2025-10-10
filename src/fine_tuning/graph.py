import json
import os
from train import load_config
import matplotlib.pyplot as plt


cfg = load_config("qwen.yaml")
log_path = os.path.join(cfg["output"]["dir"], "training_log.json")

with open(log_path) as f:
    logs = json.load(f)

train_steps, train_loss, eval_steps, eval_loss = [], [], [], []

for entry in logs:
    if "loss" in entry and "step" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    if "eval_loss" in entry and "step" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])

plt.plot(train_steps, train_loss, label="Training Loss")
plt.plot(eval_steps, eval_loss, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
