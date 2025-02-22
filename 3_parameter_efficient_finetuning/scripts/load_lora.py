#!/usr/bin/env python3

import logging
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main function of the script."""
    setup_logging()
    logging.info(f"main() invoked")

    peft_model_id = "ybelkada/opt-6.7b-lora"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)

    batch = tokenizer("Machine learning is: ", return_tensors='pt')
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=128, do_sample=True, temperature=0.7)

    print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    logging.info("exit")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
    except Exception as e:
        logging.exception("An error occurred")
        sys.exit(1)