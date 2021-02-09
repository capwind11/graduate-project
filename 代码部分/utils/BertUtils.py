import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

class BertUtils:

    def __init__(self):
        self.idx_to_bert = []

    def convert_sen_to_bert(self):
        return

    '''
    用隐藏层倒数第二层的平均值作为句向量编码
    '''
    def sentenceEmbeddingAvg(encoded_layers):
        return torch.mean(encoded_layers[-2], 1)


    def sentenceEmbeddingAvg(encoded_layers):
        return encoded_layers[-2]


    def sentenceEmbeddingAvg(encoded_layers):
        return encoded_layers[-2][0]


    '''
    将后面四层词向量拼接后作为词向量编码
    '''
    def wordEmbeddingByConcat(token_embeddings):
        return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                token_embeddings]  # [number_of_tokens, hidden_size*4]

    '''
    将后面四层词向量相加后作为词向量编码
    '''
    def wordEmbeddingBySummed(token_embeddings):
        return [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]  # [number_of_tokens, hidden_size]