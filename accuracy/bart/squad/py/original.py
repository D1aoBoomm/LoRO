#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# In[47]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model.cuda()
model.eval()


# In[48]:


print(model)


# In[49]:


question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device='cuda')

context = 'Tom likes playing with Jerry.'
question = 'Who does Tom like to play with?'
question_answerer(question, context)


# In[50]:


import torch

question, text = 'Who does Tom like playing with?', 'Tom likes playing with Jerry.'
encoding = tokenizer(question, text, return_tensors='pt')
input_ids = encoding['input_ids'].to('cuda')
attention_mask = encoding['attention_mask'].to('cuda')

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
print(answer)


# In[51]:


dataset = load_dataset("rajpurkar/squad")['validation'] # this is example, u need to test it using official test set
print(dataset)


# In[52]:


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

