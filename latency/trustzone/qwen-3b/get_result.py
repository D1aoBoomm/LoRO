from model_info import *

with open('./result.txt', 'r') as f:
    time_dict = f.read().split('\n')[0]
    time_dict = eval(time_dict)
    
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
total_time += time_dict['out_ree']
total_ree += time_dict['out_ree']

print('--------------------------------------')
print("LoRO Total Time (Parallel between REE and TEE): {}".format(total_time))
print("LoRO Total Time (No Parallel): {}".format(total_ree + total_transfer + total_tee))
print("LoRO Time Breakdown: TEE:{} REE:{} Transfer:{}".format(total_tee, total_ree, total_transfer))
print("REE Inference Time: {}".format(total_ree))

with open('./result.txt', 'a+') as f:
    f.write('\nLoRO Total Time (Parallel between REE and TEE): {}'.format(total_time))
    f.write('\nLoRO Total Time (No Parallel): {}'.format(total_ree + total_transfer + total_tee))
    f.write('\nLoRO Time Breakdown: TEE:{} REE:{} Transfer:{}'.format(total_tee, total_ree, total_transfer))
    f.write('\nREE Inference Time: {}'.format(total_ree))