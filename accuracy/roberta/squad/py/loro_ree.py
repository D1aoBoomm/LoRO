#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append("../../../../LoRO")

from utils import *


# In[2]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("PremalMatalia/roberta-base-best-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("PremalMatalia/roberta-base-best-squad2")


# In[3]:


print(model)


# In[4]:


#obfuscate model
model = model_obfuscation(model)
print(model)


# In[5]:


model = obfus_inference_mode(model)


# In[6]:


question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device='cuda')


# In[7]:


context = 'Tom likes playing with Jerry.'
question = 'Who does Tom like playing with?'
result = question_answerer(question=question, context=context)
print(result)


# In[8]:


dataset = load_dataset("rajpurkar/squad")['validation']
print(dataset)


# In[9]:


correct = 0
total = 0

for i in tqdm.tqdm(range(10570)):
    result = question_answerer(question=dataset[i]['question'], context=dataset[i]['context'])
    for answer in dataset[i]['answers']['text']: # exact match
        if answer == result['answer']:
            correct += 1
            break
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

