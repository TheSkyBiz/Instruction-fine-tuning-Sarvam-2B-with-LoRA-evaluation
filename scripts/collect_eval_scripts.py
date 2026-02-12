import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.config import MODEL_NAME, GEN_MAX_TOKENS, GEN_TEMPERATURE

# -------- Paths --------
EVAL_DIR = "eval"
PROMPTS_PATH = os.path.join(EVAL_DIR, "prompts.json")
BEFORE_PATH = os.path.join(EVAL_DIR, "before_outputs.json")
AFTER_PATH = os.path.join(EVAL_DIR, "after_outputs.json")

ADAPTER_PATH = os.path.join(os.getcwd(), "lora_sarvam", "checkpoint-60")

os.makedirs(EVAL_DIR, exist_ok=True)

# -------- Load or init JSON files --------
def load_or_init(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

prompts = load_or_init(PROMPTS_PATH)
before_outputs = load_or_init(BEFORE_PATH)
after_outputs = load_or_init(AFTER_PATH)

# -------- Load models --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

def generate(model, prompt):
    full_prompt = f"""### Instruction:
{prompt}

### Response:
"""
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- Interactive loop --------
print("\nEnter prompts one by one. Type 'exit' to stop.\n")

while True:
    prompt = input("PROMPT >> ").strip()
    if prompt.lower() == "exit":
        break
    if not prompt:
        continue

    print("\nGenerating BEFORE (base model)...")
    before = generate(base_model, prompt)

    print("\nGenerating AFTER (LoRA model)...")
    after = generate(lora_model, prompt)

    # Save
    prompts.append(prompt)
    before_outputs.append(before)
    after_outputs.append(after)

    with open(PROMPTS_PATH, "w") as f:
        json.dump(prompts, f, indent=2)

    with open(BEFORE_PATH, "w") as f:
        json.dump(before_outputs, f, indent=2)

    with open(AFTER_PATH, "w") as f:
        json.dump(after_outputs, f, indent=2)

    print("\n--- SAVED ---")
    print("BEFORE:\n", before)
    print("\nAFTER:\n", after)
    print("\n-----------------------------\n")

print("Done collecting evaluation data.")
