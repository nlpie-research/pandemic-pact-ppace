import torch

import datasets as ds
import transformers as ts

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

import os

from peft import *

from accelerate import Accelerator

from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

huggingface_token = "YOUR_HUGGINGFACE_TOKEN"

# Load the data
pandemic_dataset = ds.load_dataset("nlpie/pandemic_pact", token=huggingface_token)["train"]

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = ts.AutoTokenizer.from_pretrained(model_id, token=huggingface_token)

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.bos_token, tokenizer.eos_token

def promptedMappingFunction(items):
  texts = []

  for index in range(len(items["promptedText"])):
    currentInput = tokenizer.apply_chat_template([{"content": items["promptedText"][index], "role": "user"}, {"content": items["output"][index], "role": "assistant"}],
                                                 tokenize=False,
                                                 add_generation_prompt=False)
    texts.append(currentInput)

  return tokenizer(texts, truncation=False, add_special_tokens=False)

promptTokenizedDataset = pandemic_dataset["train"].map(promptedMappingFunction, batched=True, remove_columns=pandemic_dataset["train"].column_names)
promptTokenizedDataset = promptTokenizedDataset.shuffle(len(promptTokenizedDataset))

n_gpus = torch.cuda.device_count()
device_map = "auto"

if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

print(device_map)

training_args = TrainingArguments(
            output_dir=f"/teamspace/studios/this_studio/llama/checkpoints/",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            bf16=True,  # Use BF16 if available
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            num_train_epochs=2,
            gradient_accumulation_steps=4,
            logging_dir=f"/content/logs/",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=10,
)

model = ts.AutoModelForCausalLM.from_pretrained(
    model_id,
    token=huggingface_token,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16)

print(model)

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]

peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

model = get_peft_model(model, peft_config)

print(model)

response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = promptTokenizedDataset,
    max_seq_length = 2048,
    data_collator = collator,
    args = training_args,
)

trainer.train()

model.save_pretrained("/teamspace/studios/this_studio/llama/final/")