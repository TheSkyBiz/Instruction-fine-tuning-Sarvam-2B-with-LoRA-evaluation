import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import MODEL_NAME
import os

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

tasks = open("data/instruction_seeds.txt").read().strip().split("\n")
dataset = []

for task in tasks:
    prompt = f"""
You are generating instruction-response pairs for training a language model.

Task: {task}

Generate 20 examples.
Format strictly as:

Instruction:
Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=800,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    blocks = text.split("Instruction:")

    for block in blocks[1:]:
        try:
            instr, resp = block.split("Response:")
            dataset.append({
                "instruction": instr.strip(),
                "response": resp.strip()
            })
        except:
            continue

output_path = "data/raw/raw_dataset.json"

# If file exists, load existing data
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        existing_data = json.load(f)
else:
    existing_data = []

# Append new samples
existing_data.extend(dataset)

# Save accumulated dataset
with open(output_path, "w") as f:
    json.dump(existing_data, f, indent=2)

print(f"Added {len(dataset)} samples")
print(f"Total samples now: {len(existing_data)}")


print(f"Generated {len(dataset)} samples")
