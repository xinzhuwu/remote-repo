import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, logging
logging.set_verbosity_error()
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")
texts=["I love multimodal information processing"]
#将句子转换成子词序列
encodes_inputs=tokenizer(texts,return_tensors="pt")
# print(encodes_inputs)
print("子词序列:", tokenizer.convert_ids_to_tokens(encodes_inputs["input_ids"][0]))
"""执行BERT前向反馈过程"""
outputs=model(**encodes_inputs)
#print(outputs)
"子词表示"
word_representation=outputs["last_hidden_state"]
"整体表示"
global_representation=torch.mean(word_representation, dim=1)
print("词表示的形状", word_representation.shape)
print("整体表示的形状", global_representation.shape)



