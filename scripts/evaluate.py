import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config.config import MODEL_NAME, GEN_MAX_TOKENS, GEN_TEMPERATURE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

ADAPTER_PATH = os.path.join(os.getcwd(), "lora_sarvam", "checkpoint-60")

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

prompt = """### Instruction:
Explain overfitting in simple terms for an Indian engineering student.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=GEN_MAX_TOKENS,
    temperature=GEN_TEMPERATURE
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

with open("results/after.txt", "w") as f:
    f.write(result)
