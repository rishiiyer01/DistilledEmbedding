

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
    return [sentence for example in batch for sentence in example['set']]

batch_size =2  
# Split the dataset into train and test
#5% of the original dataset
small_dataset = retrieval_set['train'].select(range(len(retrieval_set['train']) // 20))  # '// 20' is equivalent to 5%

train_test_split = small_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Create dataloaders for both train and test sets
trainloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

def distillation_loss(student_logits, teacher_logits, temperature=1):
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
    
    for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        # Move batch to GPU
        #input_ids = batch['input_ids'].to('cuda')
        #attention_mask = batch['attention_mask'].to('cuda')
        #batch=batch.to('cuda')
        
        # Get embeddings from original model
        with torch.no_grad():
            original_embeddings = original_model.encode(batch) #encode method built in to forward of distilled model

        # Get embeddings from distilled model
        distilled_embeddings = distilled_model(batch)
        #print(distilled_embeddings.grad_fn)
        # Compute loss
        loss = distillation_loss(distilled_embeddings, original_embeddings)
        
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    # Evaluation
    distilled_model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            batch.to('cuda')
            original_embeddings = original_model.encode(batch)
            distilled_embeddings = distilled_model(batch)
            
            loss = distillation_loss(distilled_embeddings, original_embeddings)
            eval_loss += loss.item()
    
    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_eval_loss:.4f}")



# Save the distilled model
torch.save(distilled_model.state_dict(), 'distilled_model.pth')