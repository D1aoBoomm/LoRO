#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.append("../../../../LoRO")
sys.path.append('..')

from label2id import label2id_dict

from utils import *


# In[1]:


# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from datasets import load_dataset

import tqdm

import torch
from torch.utils.data import DataLoader

processor = AutoImageProcessor.from_pretrained("nateraw/vit-base-food101")
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-food101")
model.eval()


# In[2]:


print(model)


# In[5]:


model = model_obfuscation(model, noise_mag=1)
model = obfus_inference_mode(model)


# In[6]:


dataset = load_dataset("food101")['validation']
print(dataset)


# In[7]:


classifier = pipeline("image-classification", model=model, image_processor=processor, device='cuda')
# examples from training set

img = dataset[0]['image'] # beignets 6

print(label2id_dict[classifier(img)[0]['label']])
print(dataset[0]['label'])

img = dataset[-1]['image'] # spaghetti_bolognese 90

print(label2id_dict[classifier(img)[0]['label']])
print(dataset[-1]['label'])


# In[8]:


# 定义预处理函数
def process_dataset(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

dataset.set_transform(process_dataset)


# In[9]:


batch_size = 192  # You can adjust this based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

correct = 0
total = 0

model = model.to('cuda')

with torch.no_grad():  # Disable gradient calculation for evaluation
    for batch in tqdm.tqdm(dataloader):
        pixel_values = batch['pixel_values'].to('cuda')
        labels = batch['label'].to('cuda')
        
        # Process images in batch
        batch_results = model(pixel_values=pixel_values)  # Assuming your classifier can handle batches
        
        predicted_labels = batch_results.logits.argmax(dim=-1)
        
        # Compare with ground truth
        correct += (predicted_labels == labels).sum().item()
        total += len(labels)

accuracy = correct / total
print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:7f}")

