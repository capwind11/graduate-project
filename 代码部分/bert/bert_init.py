from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForMaskedLM
import numpy as np
import torch

# 加载bert的分词器
tokenizer = BertTokenizer.from_pretrained('D:/bert_pytorch/bert-base-uncased-vocab.txt')
# 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
bert = BertModel.from_pretrained('D:/bert-pytorch/bert-base-uncased/bert-base-uncased')

# Tokenized input
text = "Who was Jim Henson ? Jim Henson was a puppeteer"
tokenized_text = tokenizer.tokenize(text)