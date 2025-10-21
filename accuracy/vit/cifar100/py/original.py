#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append('..')

from label2id import label2id_dict


# In[2]:


# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from datasets import load_dataset

import tqdm

import torch
from torch.utils.data import DataLoader

processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
model.eval()


# In[3]:


print(model)


# In[4]:


dataset = load_dataset("cifar100")['test']
print(dataset)


# In[5]:


classifier = pipeline("image-classification", model=model, image_processor=processor, device='cuda')
# examples from training set

img = dataset['img'][0] # mountain 49

print(label2id_dict[classifier(img)[0]['label']])
print(dataset[0]['fine_label'])

img = dataset['img'][1] # forest 33

print(label2id_dict[classifier(img)[0]['label']])
print(dataset[1]['fine_label'])


# In[6]:


# 定义预处理函数
def process_dataset(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x.convert('RGB') for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['fine_label'] = example_batch['fine_label']
    return inputs

dataset.set_transform(process_dataset)


# In[12]:


batch_size = 128  # You can adjust this based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0

model = model.to('cuda')

with torch.no_grad():  # Disable gradient calculation for evaluation
    for batch in tqdm.tqdm(dataloader):
        pixel_values = batch['pixel_values'].to('cuda')
        labels = batch['fine_label'].to('cuda')
        
        # Process images in batch
        batch_results = model(pixel_values=pixel_values)  # Assuming your classifier can handle batches
        
        predicted_labels = batch_results.logits.argmax(dim=-1)
        
        # Compare with ground truth
        correct += (predicted_labels == labels).sum().item()
        total += len(labels)

accuracy = correct / total
print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:7f}")

