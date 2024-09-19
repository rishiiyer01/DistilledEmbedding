#file that prunes the distilled model

import torch

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from ablatedEmbedder import AblatedEmbedder
from model import LoRALayer, LinearWithLoRA, DistilledModel

#we need to identify nodes/neurons with low activations on the dataset


class PrunedDistilledModel(nn.Module):
    def __init__(self, distilled_model, pruning_rate=0.5):
        super(PrunedDistilledModel, self).__init__()
        self.distilled_model = distilled_model
        self.pruning_rate = pruning_rate
        
        self.activations=None
    def activation_recorder(self,x):
        with torch.no_grad():
            if self.activations==None:
                self.activations=self.distilled_model(x)
            else:
    
                x=self.distilled_model(x)
                self.activations=(self.activations+x)/2
        
    def prune(self,x):
        #get 50% topk indices of self.activations
        num_to_keep = int(self.activations.shape[0] * (1 - self.pruning_rate))
        _, top_indices = torch.topk(self.activations, num_to_keep)
        return x[top_indices]
    
    def forward(self, x):
        x=self.distilled_model(x)
        pruned_embeddings=prune(x)
        return pruned_embeddings


ablation_list=[30,14,16,20,29,12,18,21,11,13,22,15,26,24,23,10]
distilled_model = DistilledModel(ablation_list).to('cuda')
distilled_model.load_state_dict(torch.load('/root/verb-workspace/DistilledEmbedding/distilled_model.pth'))
pruning_rate = 0.5  # Adjust this value to control the amount of pruning
pruned_model = PrunedDistilledModel(distilled_model, pruning_rate).to('cuda')

ds = load_dataset("nampdn-ai/tiny-textbooks")
#first we want to go through a single epoch with activation recorder
#we will be using the ['train'] set, which inside we are only using the ['text'] for the first 100000 examples/passages
#we use a high quality general small text dataset for this step
pruneset=ds['train']['text'][:100000]


#now we finetune post pruning on similarity (traditional embedding model training)
retrieval_set = load_dataset("embedding-data/WikiAnswers")
small_dataset = retrieval_set['train'].select(range(len(retrieval_set['train']) // 200)) #approx 100000 sentences

train_test_split = small_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

trainset=ds2['train']['text']
testset=ds2['test']['text']
batch_size=1 #forced to use very small batch size here because of memory consumption
def collate_fn(batch):
    
    return batch


def collate_fn2(batch):
    
    return [example['set'] for example in batch]

trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=False)
pruneloader=DataLoader(pruneset,batch_size=batch_size,collate_fn=collate_fn,shuffle=True)

for batch in pruneloader:
    pruned.model.activation_recorder(batch)



#now we finetune
num_epochs=1 #1-2 epochs for finetuning here is fine, the dataset is quite large, and we are only using a small subset of 100k sentences

def pruned_loss(embeddings, temperature=1):
    # Compute pairwise similarities within each set
    loss = 0
    for set_embeddings in embeddings:
        if len(set_embeddings) < 2:
            continue  # Skip sets with only one sentence
        
        # Normalize embeddings
        set_embeddings = F.normalize(set_embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(set_embeddings, set_embeddings.t())
        
        # Create target: all pairs should be similar (1) except self-similarity
        target = torch.ones_like(similarities) - torch.eye(similarities.shape[0]).to(similarities.device)
        
        # Compute loss (you can experiment with different loss functions)
        set_loss = F.mse_loss(similarities, target)
        loss += set_loss
    
    return loss / len(embeddings)


optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=1e-4)

try:
    for epoch in range(num_epochs):
        pruned_model.train()
        total_loss = 0
        
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            
            
           
    
            # Get embeddings from distilled model
            pruned_embeddings = [pruned_model(sentences) for sentences in batch]
            #print(distilled_embeddings.grad_fn)
            # Compute loss
            loss = pruned_loss(pruned_embeddings)
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        # Evaluation
        pruned_model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(testloader, desc="Evaluating"):
                
                
                pruned_embeddings = [pruned_model(sentences) for sentences in batch]
                
                loss = pruned_loss(pruned_embeddings)
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_eval_loss:.4f}")
except:

    torch.save(pruned_model.state_dict(), 'pruned_model.pth')

# Save the pruned model
torch.save(pruned_model.state_dict(), 'pruned_model.pth')