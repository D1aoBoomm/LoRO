import copy

import torch

from loro import LoroLinear

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/media/ubuntu/BE6A02386A01EDC9/huggingface"
os.environ['HF_HUB_CACHE'] = "/media/ubuntu/BE6A02386A01EDC9/huggingface/hub"
os.environ['HF_DATASET_HOME'] = "/media/ubuntu/BE6A02386A01EDC9/huggingface/datasets"


# 替换模型子模块
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

# 执行loro模型混淆
def model_obfuscation(original_model, device='cuda', noise_mag=1e-1):
    obfuscated_model = copy.deepcopy(original_model)
    for name, module in obfuscated_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print("Obfuscating: {}".format(name))
            _set_module(obfuscated_model, name, LoroLinear(auto_mode=True, original_linear=module, device=device, noise_mag=noise_mag))
    
    return obfuscated_model

# 把loro模型转换成不解混淆模式
def de_obfus_inference_mode(obfuscated_model):
    for name, module in obfuscated_model.named_modules():
        if isinstance(module, LoroLinear):
            module.deobfus_inference = True
    return obfuscated_model

# 把loro模型转换成解混淆模式
def obfus_inference_mode(obfuscated_model):
    for name, module in obfuscated_model.named_modules():
        if isinstance(module, LoroLinear):
            module.deobfus_inference = False
    return obfuscated_model