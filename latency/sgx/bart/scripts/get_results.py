import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from model_info import *

with open('./results/tee_time.txt', 'r') as f:
    time_dict_1 = f.read().split('\n')[0]
    time_dict_1 = eval(time_dict_1)
    
with open('./results/ree_time.txt', 'r') as f:
    time_dict_2 = f.read().split('\n')[0]
    time_dict_2 = eval(time_dict_2)

time_dict = {**time_dict_1, **time_dict_2}
print(time_dict)

total_time = 0
total_tee = 0
total_ree = 0
total_transfer = 0

for layer in range(layers):

    # qkv
    for attention in ['q', 'k', 'v']:
        total_time += max(time_dict[f'{attention}_{layer}_tee'] + time_dict[f'{attention}_{layer}_transfer'], time_dict[f'{attention}_{layer}_ree']) 
        total_tee += time_dict[f'{attention}_{layer}_tee']
        total_ree += time_dict[f'{attention}_{layer}_ree']
        total_transfer += time_dict[f'{attention}_{layer}_transfer']
    
    # attention
    total_time += time_dict[f'self_attention_{layer}']
    total_ree += time_dict[f'self_attention_{layer}']

    # ffn
    total_time += max(time_dict[f'ffn_{layer}_ree'], time_dict[f'ffn_{layer}_transfer'] + time_dict[f'ffn_{layer}_tee'])
    total_ree += time_dict[f'ffn_{layer}_ree']
    total_transfer += time_dict[f'ffn_{layer}_transfer']
    total_tee += time_dict[f'ffn_{layer}_tee']

# output
total_time += max(time_dict['out_tee'] + time_dict['out_transfer'], time_dict['out_ree'])
total_ree += time_dict['out_ree']
total_transfer += time_dict['out_transfer']
total_tee += time_dict['out_tee']

print('--------------------------------------')
print("LoRO Total Time (Parallel between REE and TEE): {}".format(total_time))
print("LoRO Total Time (No Parallel): {}".format(total_ree + total_transfer + total_tee))
print("LoRO Time Breakdown: TEE:{} REE:{} Transfer:{}".format(total_tee, total_ree, total_transfer))
print("REE Inference Time: {}".format(total_ree))

with open('./results/result.txt', 'w') as f:
    f.write('\nLoRO Total Time (Parallel between REE and TEE): {}'.format(total_time))
    f.write('\nLoRO Total Time (No Parallel): {}'.format(total_ree + total_transfer + total_tee))
    f.write('\nLoRO Time Breakdown: TEE:{} REE:{} Transfer:{}'.format(total_tee, total_ree, total_transfer))
    f.write('\nREE Inference Time: {}'.format(total_ree))