#ablated embedding model

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List






import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List

class AblatedEmbedder(nn.Module):
    def __init__(self, ablation_layers: List[int]):
        super(AblatedEmbedder, self).__init__()
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create a new ModuleList without the ablated layers
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.embedding_model.layers)
            if i not in ablation_layers
        ])
        del self.model.embedding_model.layers
        # Replace the original layers with the new ModuleList
        self.model.embedding_model.layers = new_layers
       
        # Update the config to reflect the new number of layers
        #self.model.config.num_hidden_layers = len(new_layers)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)


abl=AblatedEmbedder([0,1])

print(abl.model)