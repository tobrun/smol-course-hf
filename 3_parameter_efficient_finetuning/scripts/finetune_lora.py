#!/usr/bin/env python3

import logging
import sys
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    pipeline
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
    AutoPeftModelForCausalLM
)

from trl import SFTConfig, SFTTrainer, setup_chat_format


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def test_inference(pipe, prompt):
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipe(
        prompt,
    )
    return outputs[0]["generated_text"][len(prompt) :].strip()

def main():
    """Main function of the script."""
    setup_logging()
    logging.info(f"main() invoked")
    dataset = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

    # Load the model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Set up the chat format
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Set our name for the finetune to be saved &/ uploaded to
    finetune_name = "SmolLM2-PEFT-LORA"
    finetune_tags = ["smol-course", "module_1"]


    # TODO: Configure LoRA parameters
    # r: rank dimension for LoRA update matrices (smaller = more compression)
    rank_dimension = 6
    # lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
    lora_alpha = 8
    # lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
    lora_dropout = 0.05

    peft_config = LoraConfig(
        r=rank_dimension,  # Rank dimension - typically between 4-32
        lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
        lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules="all-linear",  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )

    # Training configuration
    # Hyperparameters based on QLoRA paper recommendations
    args = SFTConfig(
        # Output settings
        output_dir=finetune_name,  # Directory to save model checkpoints
        # Training duration
        num_train_epochs=1,  # Number of training epochs
        # Batch size settings
        per_device_train_batch_size=2,  # Batch size per GPU
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch
        # Memory optimization
        gradient_checkpointing=True,  # Trade compute for memory savings
        # Optimizer settings
        optim="adamw_torch_fused",  # Use fused AdamW for efficiency
        learning_rate=2e-4,  # Learning rate (QLoRA paper)
        max_grad_norm=0.3,  # Gradient clipping threshold
        # Learning rate schedule
        warmup_ratio=0.03,  # Portion of steps for warmup
        lr_scheduler_type="constant",  # Keep learning rate constant after warmup
        # Logging and saving
        logging_steps=10,  # Log metrics every N steps
        save_strategy="epoch",  # Save checkpoint every epoch
        # Precision settings
        bf16=True,  # Use bfloat16 precision
        # Integration settings
        push_to_hub=True,
        report_to="none",  # Disable external logging
    )

    max_seq_length = 1512  # max sequence length for model and packing of the dataset

    # Create SFTTrainer with LoRA configuration
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        peft_config=peft_config,  # LoRA configuration
        max_seq_length=max_seq_length,  # Maximum sequence length
        tokenizer=tokenizer,
        packing=True,  # Enable input packing for efficiency
        dataset_kwargs={
            "add_special_tokens": False,  # Special tokens handled by template
            "append_concat_token": False,  # No additional separator needed
        },
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()

    # save model
    trainer.save_model()


    # Merge LoRA Adapter into the original model
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        args.output_dir, safe_serialization=True, max_shard_size="2GB"
    )

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    # Load Model with PEFT adapter
    tokenizer = AutoTokenizer.from_pretrained(finetune_name)
    model = AutoPeftModelForCausalLM.from_pretrained(
        finetune_name, device_map="auto", torch_dtype=torch.float16
    )
    pipe = pipeline(
        "text-generation", model=merged_model, tokenizer=tokenizer, device=device
    )

    prompts = [
        "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
        "Write a Python function to calculate the factorial of a number.",
        "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
        "What is the difference between a fruit and a vegetable? Give examples of each.",
    ]

    for prompt in prompts:
        print(f"    prompt:\n{prompt}")
        print(f"    response:\n{test_inference(pipe, prompt)}")
        print("-" * 50)

    logging.info("exit")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
    except Exception as e:
        logging.exception("An error occurred")
        sys.exit(1)