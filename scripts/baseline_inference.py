import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import MODEL_NAME, GEN_MAX_TOKENS, GEN_TEMPERATURE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

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

with open("results/before.txt", "w") as f:
    f.write(result)