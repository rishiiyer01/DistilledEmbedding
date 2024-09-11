

#training, distillation file

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from ablatedEmbedder import AblatedEmbedder
from model import LoRALayer, LinearWithLoRA, DistilledModel

#list taken from ablationLayerEval
#ablating half the layers
ablation_list=[30,14,16,20,29,12,18,21,11,13,22,15,26,24,23,10]

original_model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
distilled_model = DistilledModel(ablation_list).to('cuda')

tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2')


retrieval_set = load_dataset("embedding-data/WikiAnswers")


def collate_fn(batch):

    sentences = [sentence for example in batch for sentence in example['set']]
    
    # Tokenize the sentences
    encodings = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    return encodings


batch_size = 32  
dataloader = DataLoader(retrieval_set['train'], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

def distillation_loss(student_logits, teacher_logits, temperature):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean')

# Optimizer
optimizer = torch.optim.AdamW(distilled_model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5  # Adjust as needed

for epoch in range(num_epochs):
    distilled_model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        # Move batch to GPU
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        
        # Get embeddings from original model
        with torch.no_grad():
            original_embeddings = original_model(input_ids, attention_mask=attention_mask)

        # Get embeddings from distilled model
        distilled_embeddings = distilled_model(input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = distillation_loss(distilled_embeddings, original_embeddings)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the distilled model
torch.save(distilled_model.state_dict(), 'distilled_model.pth')