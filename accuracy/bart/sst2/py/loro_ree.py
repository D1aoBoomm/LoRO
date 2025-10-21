#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append("../../../../LoRO")

from utils import *


# In[11]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
model = AutoModelForSequenceClassification.from_pretrained("valhalla/bart-large-sst2")


# In[12]:


print(model)


# In[14]:


#obfuscate model
model = model_obfuscation(model)
print(model)


# In[15]:


model = obfus_inference_mode(model)


# In[16]:


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')


# In[17]:


# two examples from training set

sentence = "contains no wit , only labored gags"

print(classifier(sentence))

sentence = "are more deeply thought through than in most ` right-thinking ' films"
print(classifier(sentence))


# In[18]:


dataset = load_dataset("glue", "sst2")['validation']
print(dataset)


# In[19]:


correct = 0
total = 0

for i in tqdm.tqdm(range(872)):
    result = classifier(dataset[i]['sentence'])
    result = 0 if result[0]['label'] == 'NEGATIVE' else 1
    if result == dataset[i]['label']:
        correct += 1
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

