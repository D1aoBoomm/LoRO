#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# In[8]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/roberta-base-qnli")
model = AutoModelForSequenceClassification.from_pretrained("JeremiahZ/roberta-base-qnli")


# In[9]:


print(model)


# In[15]:


classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')


# In[ ]:


# two examples from training set

question = "When did the third Digimon series begin?"
sentence = "Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese."

print(classifier(question+" "+sentence))

question = "What two things does Popper argue Tarski's theory involves in an evaluation of truth?"
sentence = "He bases this interpretation on the fact that examples such as the one described above refer to two things: assertions and the facts to which they refer."
print(classifier(question+" "+sentence))


# In[24]:


dataset = load_dataset("glue", "qnli")['validation']
print(dataset)


# In[28]:


correct = 0
total = 0

for i in tqdm.tqdm(range(5463)):
    result = classifier(dataset[i]['question'] + " " + dataset[i]['sentence'])
    result = 0 if result[0]['label'] == 'entailment' else 1
    if result == dataset[i]['label']:
        correct += 1
    total += 1

print("correct:{}, total:{}, accuracy:{}".format(correct, total, correct/total))

