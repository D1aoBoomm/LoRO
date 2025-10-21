#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# In[2]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("Intel/bart-large-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("Intel/bart-large-mrpc")


# In[3]:


print(model)


# In[4]:


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')


# In[5]:


# two examples from training set

sentence = "Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence . Referring to him as only \" the witness \" , Amrozi accused his brother of deliberately distorting his evidence ." # 1 equaivalent

print(classifier(sentence))

sentence = "Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion . Yucaipa bought Dominick 's in 1995 for $ 693 million and sold it to Safeway for $ 1.8 billion in 1998 ." # 0 not_equivalent

print(classifier(sentence))


# In[6]:


dataset = load_dataset("glue", "mrpc")['validation']
print(dataset)


# In[7]:


correct = 0
total = 0

for i in tqdm.tqdm(range(408)):
    result = classifier(dataset[i]['sentence1'] + ' ' + dataset[i]['sentence2'])

    if result[0]['label'] == 'equivalent':
        result = 1
    elif result[0]['label'] == 'not_equivalent':
        result = 0
    else:
        exit()
        
    if result == dataset[i]['label']:
        correct += 1
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

