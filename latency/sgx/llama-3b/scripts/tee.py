import numpy as np
import tqdm

import time

from model_info import * # get model structure

# lib_transfer = ctypes.cdll.LoadLibrary('../our_c_lib/data_transfer/libtransfer.so')

# transfer_func = lib_transfer.transfer
# transfer_func.argtypes = [ctypes.c_int, ctypes.c_int]
# transfer_func.restype = ctypes.c_float

def matrix_multiplication(s, h, output_h, bias=True):
    x = np.random.rand(s, h)
    w = np.random.rand(h, output_h)
    b = np.random.rand(output_h)
    
    if bias == True:
        start = time.time()
        result = x@w
        end = time.time()
    else:
        start = time.time()
        result = x@w + b
        end = time.time()
    return end-start, result

time_dict = {}

# warmup
print('warming up...')
for i in range(10):
    matrix_multiplication(s, h, r, True)
    matrix_multiplication(s, r, h, True)

print('starting inference...')
for layer in tqdm.tqdm(range(layers)):

    # QKV
    for attention in ['q','k','v']:
        if attention in ['k', 'v']:
            h_now = h // group_num
        else:
            h_now = h
        
        # inference
        tee_time_1 = matrix_multiplication(s, h, r, bias=False)[0]
        tee_time_2 = matrix_multiplication(s, r, h_now, bias=False)[0]
        
        # otp
        x = np.random.rand(s,h_now)
        start = time.time()
        x = x+x
        end = time.time()
        
        time_dict["{}_{}_tee".format(attention, layer)] = tee_time_1 + tee_time_2 + end-start

    # FFN
    tee_time_ffn_1 = matrix_multiplication(s, h, r, bias=False)[0] + matrix_multiplication(s, r, ffn_h, bias=False)[0]
    tee_time_ffn_2 = matrix_multiplication(s, ffn_h, r, bias=False)[0] + matrix_multiplication(s, r, h, bias=False)[0]
    time_dict["ffn_{}_tee".format(layer)] = tee_time_ffn_1 + tee_time_ffn_2

# Output Layer
tee_time_out_1 = matrix_multiplication(s, h, r, bias=False)[0] + matrix_multiplication(s, r, h, bias=False)[0]
tee_time_out_2 = matrix_multiplication(s, h, r, bias=False)[0] + matrix_multiplication(s, r, classification_num, bias=False)[0]
time_dict["out_tee"] = tee_time_out_1 + tee_time_out_2

with open('./results/tee_time.txt', 'w') as f:
    f.write(str(time_dict)   
)
print('saved to result.txt')   
