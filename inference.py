import torch

import datasets as ds
import transformers as ts

import os

pandemic_dataset = ds.load_dataset("nlpie/pandemic_pact")["validation"]

def chatMappingFunction(items):
  outputs = {
      "messages": [],
  }

  for index in range(len(items["promptedText"])):
    currentOutput = [
        {"content": items["promptedText"][index], "role": "user"}, 
    ]

    outputs["messages"].append(currentOutput)
    

  return outputs

conversationDataset = pandemic_dataset.map(chatMappingFunction, batched=True)
# conversationDataset = conversationDataset.filter(lambda x: ',' in x["categories"])
print(conversationDataset)

model_id = "nlpie/ppace-v1.0"

tokenizer = ts.AutoTokenizer.from_pretrained(model_id)

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.bos_token, tokenizer.eos_token

model = ts.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16)


print(model)

def generateOutput(inputs):
  inputs = tokenizer(
  [
      tokenizer.decode(tokenizer.apply_chat_template(inputs))
  ], return_tensors = "pt").to("cuda")

  from transformers import TextStreamer
  text_streamer = TextStreamer(tokenizer, skip_prompt=True)
  output = model.generate(**inputs, max_new_tokens = 512, num_beams=4, eos_token_id=tokenizer("<|eot_id|>", add_special_tokens=False)["input_ids"][0])

  return tokenizer.decode(output[0])

index = 100

inputs = [conversationDataset[index]["messages"][0]]
print(inputs[0]["content"])
print()

print("started processing")

output = generateOutput(inputs)
print(output)
print()
print()
print("Categories Are: " + str(conversationDataset[index]["categories"]))
