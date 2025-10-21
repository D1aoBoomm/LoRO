#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/media/ubuntu/BE6A02386A01EDC9/huggingface"
os.environ['HF_HUB_CACHE'] = "/media/ubuntu/BE6A02386A01EDC9/huggingface/hub"
os.environ['HF_DATASET_HOME'] = "/media/ubuntu/BE6A02386A01EDC9/huggingface/datasets"

import timeout_decorator
from timeout_decorator import TimeoutError

import sys
sys.path.append("../../../../LoRO")
sys.path.append('..')

from utils import *


# In[2]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("Creekside/Qwen-3B-gsm8k-GRPO")
model = AutoModelForCausalLM.from_pretrained("Creekside/Qwen-3B-gsm8k-GRPO")


# In[3]:


print(model)


# In[4]:


model = model_obfuscation(model)
print(model)


# In[5]:


dataset = load_dataset("GSM8K","main")['test']
print(dataset)


# In[7]:


question_answerer = pipeline("text-generation", model=model, tokenizer=tokenizer)


# In[8]:


idx = 3
question = dataset[idx]['question']

expected_answer =  dataset[idx]['answer']

prompt = [{ "role" : "user" , "content" : '{} Please think step by step, give the final number in ONE new line after ####, without other words. Your answer will be considered wrong if not follow this rule.'}]
print(prompt)
print(expected_answer)
print(prompt[0]['content'].format(question))


# In[9]:


query = prompt
query[0]['content'] = query[0]['content'].format(question)
answer = question_answerer(query, do_sample=True)
# print(answer[0]['generated_text'][1]['content'])
print(answer[0]['generated_text'][1]['content'].split('####')[-1])


# In[9]:


timeout_seconds  = 60

@timeout_decorator.timeout(timeout_seconds, timeout_exception=TimeoutError)
def inference(pipe, query):
    return pipe(query, do_sample=False)

correct = 0
total = 0

model.eval()

progress_bar = tqdm.tqdm(range(len(dataset)))

for i in progress_bar:
    
    total += 1
    
    question = dataset[i]['question']
    expected_answer =  dataset[i]['answer'].split('#### ')[-1]
    
    query = [{ "role" : "user" , "content" : '{} Please think step by step, give the final number in a new line after #### without any other words.'.format(question)}]
    try:
        answer = inference(question_answerer, query)[0]['generated_text'][1]['content'].split('####')[-1].strip()
    except:
        print("TimeoutError index:{}".format(i))

    if answer == expected_answer:
        correct += 1
    
    progress_bar.set_postfix({'correct': correct, 'total': total, 'acc': correct/total})

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

