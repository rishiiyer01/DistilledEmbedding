#measuring the importance of layers
#some prior research:
#in the distilbert paper, they suggest depth reduction is more important than embed_dim reduction
#some other papers contradict this
#it is well known that first and last layers are most important


import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List

class AblationLayerEstimator(nn.Module):
    def __init__(self, ablation_layer: int):
        super(AblationLayerEstimator, self).__init__()
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True,torch_dtype=torch.float8_e4m3fn).to('cuda')
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set the parameters of the specified layer to zero
        layer = self.model.embedding_model.layers[ablation_layer]
        for param in layer.parameters():
            param.data.zero_()

    def forward(self, x):
        return self.model.encode(x)

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            with torch.no_grad():
                embeddings = self.forward(batch)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)