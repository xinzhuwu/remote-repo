import torch
import torch.nn as nn
rnn=nn.GRU(input_size=50, hidden_size=64,  num_layers=2, batch_first=True, bidirectional=False)
input=torch.randn(1,5,50)
output, hidden=rnn(input)
print("Output形状:",output.shape)
print("Hidden形状:" , hidden.shape)
word_representation=output
global_representation_lastword=hidden[-1]
# 也可以是基于output global_representation_lastword=output[:,-1,:]
"""torch.max默认返回结果是(values, indicies),  torch.amax默认值只返回最大值，不返回索引"""
global_representation_avg=torch.mean(output, dim=1)
global_representation_max=torch.amax(output, dim=1)
print("词表示的形状:", word_representation.shape)
print("整体表示(最后一个词的表示)形状:", global_representation_lastword.shape)
print("整体表示(所有词表示的平均值)形状:", global_representation_avg.shape)
print("整体表示(所有词表示的最大值)形状", global_representation_max.shape)


