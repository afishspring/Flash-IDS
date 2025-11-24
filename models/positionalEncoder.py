import math
import numpy as np
import torch

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]
    

class DocumentInferrer:
    def __init__(self, encoder, w2vmodel):
        self.encoder = encoder
        self.w2vmodel = w2vmodel

    def __call__(self, document):
        # 使用 self.w2vmodel 和 self.encoder
        word_embeddings = [self.w2vmodel.wv[word] for word in document if word in self.w2vmodel.wv]
        
        if not word_embeddings:
            return np.zeros(20)

        output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
        
        if len(document) < 100000:
            output_embedding = self.encoder.embed(output_embedding)

        output_embedding = output_embedding.detach().cpu().numpy()
        return np.mean(output_embedding, axis=0)
