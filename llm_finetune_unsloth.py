# -*- coding: utf-8 -*-
"""llm_finetune_unsloth.py

Fine-tune Llama 3.2 3B model using Unsloth
"""

import sys
import subprocess
import pkg_resources

# Check and install required packages
required_packages = [
    'torch',
    'transformers',
    'datasets',
    'accelerate',
    'trl',
    'peft',
    'bitsandbytes',
    'scipy'
]

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ Installed {package}")
        return True
    except:
        print(f"✗ Failed to install {package}")
        return False

print("Checking and installing required packages...")
for package in required_packages:
    try:
        pkg_resources.require(package)
        print(f"✓ {package} is already installed")
    except:
        install_package(package)

# Now import packages
import torch
import os

# Check for GPU availability
print("\n" + "="*50)
print("Hardware and CUDA Check")
print("="*50)

has_cuda = torch.cuda.is_available()
print(f"CUDA Available: {has_cuda}")
if has_cuda:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Warning: No GPU detected. Training will be very slow on CPU!")
    print("Consider using Google Colab or a cloud GPU instance for better performance.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Try to install unsloth with appropriate version
try:
    import unsloth
    print("✓ Unsloth is already installed")
except ImportError:
    print("Installing Unsloth...")
    # Install unsloth based on CUDA availability
    if has_cuda:
        # Try to install the CUDA version
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[colab-new]"])
            print("✓ Installed Unsloth with CUDA support")
        except:
            print("Trying basic Unsloth installation...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth"])
    else:
        # For CPU only
        print("Installing CPU-compatible version of Unsloth...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[cpu]"])

# Now import unsloth and other libraries
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings('ignore')

# Configuration
print("\n" + "="*50)
print("Configuration Setup")
print("="*50)

max_seq_length = 1024  # Reduced for memory constraints
dtype = None

# Adjust settings based on GPU availability
if has_cuda:
    load_in_4bit = True
    print("Using 4-bit quantization (requires GPU)")
else:
    load_in_4bit = False
    print("Running in CPU mode - no quantization")
    # Use smaller model for CPU
    model_name = "unsloth/Llama-3.2-1B-Instruct"  # Smaller model for CPU
    print(f"Using smaller model for CPU: {model_name}")

if has_cuda:
    model_name = "unsloth/Llama-3.2-3B-Instruct"

print(f"Model: {model_name}")
print(f"Max sequence length: {max_seq_length}")

# Load model and tokenizer
try:
    print("\nLoading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map="auto" if has_cuda else None,
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    # Alternative loading for CPU
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if not has_cuda else torch.float16,
        device_map="auto" if has_cuda else None,
        low_cpu_mem_usage=True,
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Prepare model for LoRA fine-tuning
print("\nPreparing model for LoRA fine-tuning...")
if 'FastLanguageModel' in globals():
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,  # Smaller rank for memory efficiency
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth" if has_cuda else False,
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )
    except:
        # Fallback to standard PEFT
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
else:
    # Use standard PEFT
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

print("✓ Model prepared for LoRA")

# Load dataset
print("\nLoading dataset...")
try:
    # Try to load the dataset
    dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", 'v0', split="train[:100]")  # Small subset for testing
    print(f"✓ Loaded dataset with {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating a small dummy dataset for testing...")
    
    # Create a dummy dataset
    from datasets import Dataset
    dummy_data = {
        "problem": ["What is 2+2?", "What is the capital of France?", "How many sides does a triangle have?"],
        "reannotated_assistant_content": [
            "Let's think step by step... 2 plus 2 equals 4.",
            "The capital of France is Paris.",
            "A triangle has 3 sides."
        ],
        "solution": ["4", "Paris", "3"]
    }
    dataset = Dataset.from_dict(dummy_data)

print("\nSample dataset entries:")
for i in range(min(2, len(dataset))):
    print(f"\nSample {i+1}:")
    print(f"Problem: {dataset[i]['problem']}")
    print(f"Thought: {dataset[i]['reannotated_assistant_content'][:100]}...")
    print(f"Solution: {dataset[i]['solution']}")

# Define prompt template
r1_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{}
</problem>

{}
{}
"""
EOS_TOKEN = tokenizer.eos_token

# Formatting function for dataset
def formatting_prompts_func(examples):
    problems = examples["problem"]
    thoughts = examples["reannotated_assistant_content"]
    solutions = examples["solution"]
    texts = []
    
    for problem, thought, solution in zip(problems, thoughts, solutions):
        text = r1_prompt.format(problem, thought, solution) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

# Apply formatting to dataset
print("\nFormatting dataset...")
dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"✓ Dataset formatted")
print(f"Sample formatted text (first 200 chars): {dataset[0]['text'][:200]}...")

# Set up training arguments
print("\n" + "="*50)
print("Setting up training configuration")
print("="*50)

# Adjust batch size based on available memory
if has_cuda:
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
else:
    per_device_train_batch_size = 1  # Very small for CPU
    gradient_accumulation_steps = 1
    print("Warning: CPU training will be very slow!")
    print("Consider using fewer steps or a smaller dataset.")

training_args = TrainingArguments(
    output_dir="./llama-3.2-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=5,
    max_steps=20,  # Reduced for testing
    learning_rate=2e-4,
    fp16=has_cuda and not is_bfloat16_supported(),
    bf16=has_cuda and is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit" if has_cuda else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=has_cuda,
)

# Set up trainer
print("\nSetting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if has_cuda else None,
    ),
)

print("✓ Trainer configured")
print(f"Total training steps: {training_args.max_steps}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")

# Train the model
print("\n" + "="*50)
print("Starting Training")
print("="*50)

try:
    trainer.train()
    print("\n✓ Training completed successfully!")
    
    # Save the model
    print("\nSaving model...")
    save_path = "./llama-3.2-finetuned-model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✓ Model saved to {save_path}")
    
except Exception as e:
    print(f"\nError during training: {e}")
    print("Saving model checkpoint...")
    try:
        trainer.save_model("./checkpoint")
        print("Model saved to checkpoint")
    except:
        print("Could not save checkpoint")

# Test inference
print("\n" + "="*50)
print("Testing Inference")
print("="*50)

try:
    # Load the saved model for inference
    if os.path.exists("./llama-3.2-finetuned-model"):
        print("Loading saved model for inference...")
        if 'FastLanguageModel' in globals():
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="./llama-3.2-finetuned-model",
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map="auto" if has_cuda else None,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-finetuned-model")
            model = AutoModelForCausalLM.from_pretrained(
                "./llama-3.2-finetuned-model",
                device_map="auto" if has_cuda else None,
                torch_dtype=torch.float32 if not has_cuda else torch.float16,
            )
    
    # Prepare test prompt
    test_question = "What is 5 + 3?"
    system_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
<problem>
{}
</problem>
"""
    
    prompt = system_prompt.format(test_question)
    
    # Apply chat template if available
    try:
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    except:
        print("Using default tokenizer without chat template")
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    
    # Encode
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    # Generate
    print(f"\nTest Question: {test_question}")
    print("\nGenerating response...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("Model Response:")
    print("="*50)
    print(response)
    
except Exception as e:
    print(f"Error during inference: {e}")
    print("\nYou can still use the saved model with:")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print('model = AutoModelForCausalLM.from_pretrained("./llama-3.2-finetuned-model")')

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print("\nSummary:")
print(f"- Device used: {device}")
print(f"- Model: {model_name}")
print(f"- Dataset size: {len(dataset)}")
print(f"- Training steps completed: {training_args.max_steps}")
print(f"- Model saved to: ./llama-3.2-finetuned-model")
print("\nTo use the trained model:")
print("from transformers import AutoModelForCausalLM, AutoTokenizer")
print('tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-finetuned-model")')
print('model = AutoModelForCausalLM.from_pretrained("./llama-3.2-finetuned-model")')




# # -*- coding: utf-8 -*-
# """llm_finetune_unsloth.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/19t8Jq3--nin_CvTnZVpNwkaV22IwAhjU
# """

# # Install required packages

# # Import all required libraries at the top
# from unsloth import FastLanguageModel
# from unsloth import is_bfloat16_supported
# from unsloth.chat_templates import get_chat_template
# from datasets import load_dataset
# from trl import SFTTrainer
# from transformers import TrainingArguments, DataCollatorForSeq2Seq
# import torch

# # Configuration
# max_seq_length = 2048
# dtype = None  # Automatically figure out which is apt
# load_in_4bit = True

# # Load model and tokenizer
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Llama-3.2-3B-Instruct",
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit
# )

# # Prepare model for LoRA fine-tuning
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,  # LoRA rank
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                     "gate_proj", "up_proj", "down_proj"],
#     lora_alpha=16,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=3407,
#     use_rslora=False,
#     loftq_config=None
# )

# # Load dataset
# dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", 'v0', split="train")

# # Display sample data
# print("Sample dataset entries:")
# print(dataset[:3])

# # Define prompt template
# r1_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
# <problem>
# {}
# </problem>

# {}
# {}
# """
# EOS_TOKEN = tokenizer.eos_token

# # Formatting function for dataset
# def formatting_prompts_func(examples):
#     problems = examples["problem"]
#     thoughts = examples["reannotated_assistant_content"]
#     solutions = examples["solution"]
#     texts = []
    
#     for problem, thought, solution in zip(problems, thoughts, solutions):
#         text = r1_prompt.format(problem, thought, solution) + EOS_TOKEN
#         texts.append(text)
    
#     return {"text": texts}

# # Apply formatting to dataset
# dataset = dataset.map(formatting_prompts_func, batched=True)
# print(f"Dataset size: {len(dataset)}")
# print(f"Sample formatted text: {dataset[0]['text'][:200]}...")

# # Set up trainer
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=2,
#     packing=False,
#     args=TrainingArguments(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         warmup_steps=5,
#         max_steps=60,
#         learning_rate=2e-4,
#         fp16=not is_bfloat16_supported(),
#         bf16=is_bfloat16_supported(),
#         logging_steps=1,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=3407,
#         output_dir="outputs",
#         report_to="none",
#         save_strategy="steps",
#         save_steps=30,
#     ),
#     data_collator=DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         padding=True,
#     ),
# )

# # Train the model
# print("Starting training...")
# trainer.train()
# print("Training completed!")

# # Save the trained model
# model.save_pretrained("llama-3.2-3b-deepseek-r1-sft")
# tokenizer.save_pretrained("llama-3.2-3b-deepseek-r1-sft")
# print("Model saved locally!")

# # Save in GGUF format
# model.save_pretrained_gguf("llama-3.2-3b-deepseek-r1-sft-GGUF", tokenizer)
# print("Model saved in GGUF format!")

# # Inference on fine-tuned model
# print("\n" + "="*50)
# print("Testing inference on fine-tuned model...")
# print("="*50)

# # Set up chat template
# sys_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.
# <problem>
# {}
# </problem>
# """

# # Test question
# test_question = "How many 'r's are present in 'strawberry'?"
# message = sys_prompt.format(test_question)

# # Apply chat template
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="llama-3.1",
# )

# # Enable faster inference
# FastLanguageModel.for_inference(model)

# # Prepare messages
# messages = [
#     {"role": "user", "content": message},
# ]

# # Tokenize input
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt",
# ).to("cuda")

# # Generate response
# print(f"\nQuestion: {test_question}")
# print("\nGenerating response...")
# with torch.no_grad():
#     outputs = model.generate(
#         input_ids=inputs,
#         max_new_tokens=1024,
#         use_cache=True,
#         temperature=1.5,
#         min_p=0.1,
#         do_sample=True,
#     )

# # Decode and print response
# response = tokenizer.batch_decode(outputs)[0]
# print("\n" + "="*50)
# print("Model Response:")
# print("="*50)

# # Clean up the response to show only the assistant's output
# try:
#     # Extract assistant's response after the last user message
#     if "assistant" in response.lower():
#         parts = response.split("assistant")
#         if len(parts) > 1:
#             print(parts[-1].strip())
#         else:
#             print(response)
#     else:
#         print(response)
# except:
#     print(response)

# print("\n" + "="*50)
# print("Inference completed!")
# print("="*50)