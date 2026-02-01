import torch
import torch.nn as nn
rnn=nn.GRU(num_layers=2, input_size=50, hidden_size=32, batch_first=True, bidirectional=True)
input=torch.randn(2,5,50)
output, hidden=rnn(input)
print("Output形状:",output.shape)
print("Hidden形状:" , hidden.shape)
word_representation_avg=(output[:,:,:output.size(2)//2] + output[:,:,:output.size(2)//2] ) / 2
word_representation_concat=output
global_representation_firstword=output[:,0,:] 
global_representation_lastword=output[:,-1,:]
print("词表示(前向和后向表示平均值)的形状:", word_representation_avg.shape)
print("词表示(前向和后向拼接)的形状:", word_representation_concat.shape)
print("整体表示(最后一个词的表示)形状:", global_representation_lastword.shape)
print("整体表示(第一个词的表示)形状:", global_representation_firstword.shape)




