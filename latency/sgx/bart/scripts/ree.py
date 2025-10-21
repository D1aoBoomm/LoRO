import ctypes
import torch
import tqdm

import timeit
import time
import json

from model_info import * # get model structure

lib_transfer = ctypes.cdll.LoadLibrary('../our_c_lib/data_transfer/libtransfer.so')

transfer_func = lib_transfer.transfer
transfer_func.argtypes = [ctypes.c_int, ctypes.c_int]
transfer_func.restype = ctypes.c_float

def ree_matmul_process(x, w, b):
    with torch.inference_mode():
        if b is not None:
            start = time.time()
            result = x@w + b
            torch.cuda.current_stream().synchronize()
            end = time.time()
            cost_time = end - start
        else:
            start = time.time()
            result = x@w
            torch.cuda.current_stream().synchronize()
            end = time.time()
    
    cost_time = end - start
    
    return cost_time, result

time_dict = {}

x = torch.rand(s, h).to('cuda')

# warmup
print('warming up...')
for i in range(10):
    w = torch.rand(h, output_h).to('cuda')
    b = torch.rand(output_h).to('cuda')
    ree_matmul_process(x, w, b)

print('starting inference...')
for layer in tqdm.tqdm(range(layers)):

    # QKV
    for attention in ['q','k','v']:
        w = torch.rand(h, output_h).to('cuda')
        b = torch.rand(output_h).to('cuda')
        
        # inference
        ree_time, y_ree = ree_matmul_process(x, w, b)


        # data_switching_time
        data_transfer_time = transfer_func(s, h)/1000

        # merge results
        start = time.time()
        y_ree = y_ree + y_ree
        torch.cuda.current_stream().synchronize()
        end = time.time()
        
        time_dict["{}_{}_ree".format(attention, layer)] = ree_time + (end - start)
        time_dict["{}_{}_transfer".format(attention, layer)] = data_transfer_time

    # self-attention
    mha = torch.nn.MultiheadAttention(embed_dim = h, num_heads = mutihead_num, dropout = 0.2).to('cuda')
    #mha = torch.compile(mha, mode = 'reduce-overhead')
    
    q = torch.rand(s, h).to('cuda')
    k = torch.rand(s, h).to('cuda')
    v = torch.rand(s, h).to('cuda')

    start = time.time()
    atten = mha(query=q, key=k, value=v)
    torch.cuda.current_stream().synchronize()
    end = time.time()

    time_dict["self_attention_{}".format(layer)] = end - start

    x = torch.rand(s, h).to('cuda')
    start = time.time()
    x = x+x
    end = time.time()
    time_dict["self_attention_{}".format(layer)] = end - start

    # attention output
    start = time.time()
    x = torch.rand(s, h).to('cuda')
    w = torch.rand(h, h).to('cuda')
    b = torch.rand(h).to('cuda')
    y = x@w+b
    torch.layer_norm(x, normalized_shape=x.shape)
    torch.cuda.current_stream().synchronize()
    end = time.time()

    time_dict["self_attention_{}".format(layer)] += end - start

    # FFN
    x = torch.rand(s, h).to('cuda')
    w_ffn_1 = torch.rand(h, ffn_h).to('cuda')
    b_ffn_1 = torch.rand(ffn_h).to('cuda')
    w_ffn_2 = torch.rand(ffn_h, h).to('cuda')
    b_ffn_2 = torch.rand(h).to('cuda')
    act_fn = torch.nn.GELU()

    start = time.time()
    x_res = x
    x = x@w_ffn_1+b_ffn_1
    x = act_fn(x)
    x = x@w_ffn_2+b_ffn_2 + x_res
    torch.layer_norm(x, normalized_shape=x.shape)
    torch.cuda.current_stream().synchronize()
    end = time.time()

    time_dict["ffn_{}_ree".format(layer)] = end - start

    time_ffn_transfer_1 = transfer_func(s, output_h)/1000
    time_ffn_transfer_2 = transfer_func(s, h)/1000

    time_dict["ffn_{}_transfer".format(layer)] = time_ffn_transfer_1 + time_ffn_transfer_2

# Output Layer
x = torch.rand(s, h).to('cuda')
w_out_1 = torch.rand(h, h).to('cuda')
b_out_1 = torch.rand(h).to('cuda')
w_out_2 = torch.rand(h, classification_num).to('cuda')
b_out_2 = torch.rand(classification_num).to('cuda')

start = time.time()
x = x@w_out_1+b_out_1
x = x@w_out_2+b_out_2
torch.cuda.current_stream().synchronize()
end = time.time()

time_dict["out_ree"] = end - start

time_out_transfer_1 = transfer_func(s, h)/1000
time_out_transfer_2 = transfer_func(s, classification_num)/1000
time_dict["out_transfer"] = time_out_transfer_1 + time_out_transfer_2

with open('./results/ree_time.txt', 'w') as f:
    f.write(str(time_dict)   
)
print('saved to result.txt')   