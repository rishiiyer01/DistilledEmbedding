#ablated embedding model

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding

from typing import List, Union, Dict, Mapping, Optional, Tuple, TypedDict
import inspect

#copied from huggingface nvidia-embed-v2 source
def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if always_add_eos:
        examples['input_texts'] = [instruction + input_example + tokenizer.eos_token for input_example in examples['input_texts']]
    batch_dict = tokenizer(
        examples['input_texts'],
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True)
    return batch_dict

#there is a lot of debugging code currently

class AblatedEmbedder(nn.Module):
    def __init__(self, ablation_layers: List[int]):
        super(AblatedEmbedder, self).__init__()
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
        #print(inspect.getsource(self.model.__class__))
        #print(self.model.config)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        #print(self.model)
        # Create a new ModuleList without the ablated layers
        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.embedding_model.layers)
            if i not in ablation_layers
        ])
        del self.model.embedding_model.layers
        # Replace the original layers with the new ModuleList
        self.model.embedding_model.layers = new_layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Update the config to reflect the new number of layers
        self.model.config.text_config.num_hidden_layers = len(new_layers)
        self.model.embedding_model.config.use_cache = False
        self.model.encode=self.encode
        
    def forward(self,x,attention_mask=None, *args, **kwargs):
        
        return self.model(x)

    def encode(self, prompts: List[str], instruction: str="", max_length: int=4096, **kwargs):
        if self.model.padding_side == "right" and self.model.is_mask_instruction == True and len(instruction) > 0:
            instruction_lens = len(self.model.tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0
        
        device = next(self.model.embedding_model.parameters()).device
        batch_dict =input_transform_func(self.model.tokenizer,
                                          {"input_texts": [prompt for prompt in prompts]},
                                          always_add_eos=True,
                                          max_length=max_length,
                                          instruction=instruction)
        features: NVEmbedFeatures = self.model.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=device)
        return self.model(**features)["sentence_embeddings"].squeeze(1)

    
