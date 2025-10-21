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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/roberta-base-sst2")
model = AutoModelForSequenceClassification.from_pretrained("JeremiahZ/roberta-base-sst2")


# In[3]:


print(model)


# In[4]:


#obfuscate model
model = model_obfuscation(model)
print(model)


# In[5]:


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')


# In[6]:


# two examples from training set

sentence = "contains no wit , only labored gags"

print(classifier(sentence))

sentence = "are more deeply thought through than in most ` right-thinking ' films"
print(classifier(sentence))


# In[7]:


dataset = load_dataset("glue", "sst2")['validation']
print(dataset)


# In[8]:


correct = 0
total = 0

for i in tqdm.tqdm(range(872)):
    result = classifier(dataset[i]['sentence'])
    result = 0 if result[0]['label'] == 'negative' else 1
    if result == dataset[i]['label']:
        correct += 1
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

