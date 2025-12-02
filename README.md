# Llama-3.2-3B DeepSeek R1 SFT

A fine-tuned Llama-3.2-3B-Instruct model trained on DeepSeek R1-style reasoning datasets.

## Model Details
- **Base Model**: `unsloth/Llama-3.2-3B-Instruct`
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Dataset**: ServiceNow-AI/R1-Distill-SFT
- **Context Length**: 2048 tokens
- **Training Steps**: 60

## Intended Use
This model is specialized for:
- Reflective, iterative reasoning (R1-style)
- Mathematical problem-solving
- Step-by-step explanation generation
- Educational assistance

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "muhammed-afsal-p-m/llama-3.2-3b-deepseek-r1-sft"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Example prompt
prompt = "You are a reflective assistant. <problem>If a train travels at 60 km/h for 2 hours, how far does it travel?</problem>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))