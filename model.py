#model file




import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List

from ablatedEmbedder import AblatedEmbedder
from functools import partial


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev).to(torch.bfloat16)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim)).to(torch.bfloat16)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

lora_r=8
lora_alpha=32
assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

class DistilledModel(nn.Module):
    def __init__(self,ablated_layers: List[int]):
        super(DistilledModel, self).__init__()
        #original params already have no grad
        self.embedder = AblatedEmbedder(ablated_layers)
        

        #all linear layers to LoRA layers
        for layer in self.embedder.model.embedding_model.layers:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
            layer.self_attn.o_proj = assign_lora(layer.self_attn.o_proj)
            layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj)
            layer.mlp.down_proj = assign_lora(layer.mlp.down_proj)
            layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)


        cross_attn_block=self.embedder.model.latent_attention_model.cross_attend_blocks[0]
        cross_attn_block.fn.to_q = assign_lora(cross_attn_block.fn.to_q)
        cross_attn_block.fn.to_kv = assign_lora(cross_attn_block.fn.to_kv)
        cross_attn_block.fn.to_out = assign_lora(cross_attn_block.fn.to_out)
        

    def forward(self, x, attention_mask):

        return self.embedder(x,attention_mask)

ablation_list=[30,14,16,20,29,12,18,21,11,13,22,15,26,24,23,10]
abl=DistilledModel(ablation_list)
print(abl.embedder.model.config)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')
sentences=["adafasf","blah,blah,blah"]
encoded=tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

embeddings=abl(encoded['input_ids'],encoded['attention_mask'])