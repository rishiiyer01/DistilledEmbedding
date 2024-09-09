#model file




import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List

from ablatedEmbedder import AblatedEmbedder





class DistilledModel(nn.Module):
    def __init__(self):
        super(DistilledModel, self).__init__()
        self.embedder = AblatedEmbedder()

    def forward(self, x):
        return self.embedder(x)