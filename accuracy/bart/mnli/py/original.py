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

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")


# In[3]:


print(model)


# In[4]:


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')


# In[5]:


# two examples from training set

sentence = "Conceptually cream skimming has two basic dimensions - product and geography. Product and geography are what make cream skimming work." # neutral

print(classifier(sentence))

sentence = "you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him. You lose the things to the following level if the people recall." # entailment

print(classifier(sentence))

sentence = "At the end of Rue des Francs-Bourgeois is what many consider to be the city's most handsome residential square, the Place des Vosges, with its stone and red brick facades. Place des Vosges is constructed entirely of gray marble." # contradiction
print(classifier(sentence))


# In[6]:


dataset = load_dataset("glue", "mnli")['validation_matched']
print(dataset)


# In[8]:


correct = 0
total = 0

for i in tqdm.tqdm(range(9815)):
    result = classifier(dataset[i]['premise'] + ' ' + dataset[i]['hypothesis'])

    if result[0]['label'] == 'neutral':
        result = 1
    elif result[0]['label'] == 'entailment':
        result = 0
    else:
        result = 2
        
    if result == dataset[i]['label']:
        correct += 1
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

