import json
import re
from statistics import mean

# --------- Load data ----------
with open("eval/prompts.json") as f:
    prompts = json.load(f)

with open("eval/before_outputs.json") as f:
    before_outputs = json.load(f)

with open("eval/after_outputs.json") as f:
    after_outputs = json.load(f)

assert len(prompts) == len(before_outputs) == len(after_outputs), \
    "Prompt and output files must be aligned"

N = len(prompts)

# --------- Helper functions ----------
def word_count(text):
    return len(text.split())

OVERGEN_PATTERNS = [
    r"explanation:",
    r"in this response",
    r"let me explain",
    r"as an ai",
    r"</s>"
]

def is_overgenerated(text):
    text = text.lower()
    return any(re.search(p, text) for p in OVERGEN_PATTERNS)

def is_instruction_adherent(text):
    """
    Proxy definition:
    - No meta commentary
    - No extra sections
    """
    return not is_overgenerated(text)

# --------- Metric 1: Average Response Length ----------
before_lengths = [word_count(t) for t in before_outputs]
after_lengths = [word_count(t) for t in after_outputs]

avg_len_before = mean(before_lengths)
avg_len_after = mean(after_lengths)

# --------- Metric 2: Over-generation Rate ----------
before_overgen = sum(is_overgenerated(t) for t in before_outputs)
after_overgen = sum(is_overgenerated(t) for t in after_outputs)

# --------- Metric 3: Instruction Adherence Rate ----------
before_adherent = sum(is_instruction_adherent(t) for t in before_outputs)
after_adherent = sum(is_instruction_adherent(t) for t in after_outputs)

# --------- Print report ----------
print("\n===== EVALUATION REPORT =====\n")

print("Average Response Length (words)")
print(f"  Before: {avg_len_before:.2f}")
print(f"  After : {avg_len_after:.2f}")
print(f"  Reduction: {((avg_len_before - avg_len_after) / avg_len_before) * 100:.1f}%\n")

print("Over-generation Rate")
print(f"  Before: {before_overgen}/{N} ({before_overgen/N*100:.1f}%)")
print(f"  After : {after_overgen}/{N} ({after_overgen/N*100:.1f}%)")
print(f"  Reduction: {(before_overgen - after_overgen)/N*100:.1f}%\n")

print("Instruction Adherence Rate")
print(f"  Before: {before_adherent}/{N} ({before_adherent/N*100:.1f}%)")
print(f"  After : {after_adherent}/{N} ({after_adherent/N*100:.1f}%)")
print(f"  Improvement: {(after_adherent - before_adherent)/N*100:.1f}%")

print("\n=============================\n")