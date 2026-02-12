import json

with open("data/raw/raw_dataset.json") as f:
    raw_data = json.load(f)

formatted = []

for item in raw_data:
    text = f"""### Instruction:
{item['instruction']}

### Response:
{item['response']}"""
    formatted.append({"text": text})

with open("data/processed/train_formatted.json", "w") as f:
    json.dump(formatted, f, indent=2)

print(f"Prepared {len(formatted)} training samples")
