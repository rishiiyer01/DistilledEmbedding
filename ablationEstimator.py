#measuring the importance of layers
#some prior research:
#in the distilbert paper, they suggest depth reduction is more important than embed_dim reduction
#some other papers contradict this
#it is well known that first and last layers are most important

import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class AblationEstimator(nn.Module):
    def __init__(self, num_ablation_layers, num_ablation_embed_dims=None):
        super(AblationEstimator, self).__init__()
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        for param in self.model.parameters():
            param.requires_grad = False
        print(self.model)
        

    def forward(self, x):
        pass


model=AblationEstimator(0)

        
