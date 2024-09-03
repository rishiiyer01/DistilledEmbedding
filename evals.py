from ablationEstimator import AblationLayerEstimator
import torch
from transformers import AutoTokenizer
from mteb import MTEB
import numpy as np

# List of tasks we want 
tasks = ["ArxivClusteringP2P", "ArxivClusteringS2S"]
num_layers = 32
results = {}

def custom_split(dataset):
    all_indices = list(range(len(dataset)))
    print(len(dataset))
    subset_size = 1
    subset_indices = random.sample(all_indices, subset_size)
    
    return {
        'test': subset_indices  # We're only using the 'test' split in this case
    }

for layer in range(num_layers):
    print(f"Evaluating model with layer {layer} ablated...")
    model = AblationLayerEstimator(layer).to('cuda')


    # Run evaluation with custom split, the reason for this is bc the evals in the ablation study are not as important
    # with more compute, we would run with a full eval or a larger partition
    
    # Create MTEB object with custom split and specified split
    evaluation = MTEB(tasks=tasks, task_langs=["en"], 
                      custom_split_function=custom_split,
                      split='test')  # Specify which split to use here
    
    # Run evaluation
    result = evaluation.run(
        model,
        evaluation_name=f"NV-Embed-v2_layer_{layer}_ablated",
        output_folder=f"results/layer_{layer}",
        
    )
    
    # Store results
    results[layer] = result

    # Clear CUDA cache after each layer evaluation
    torch.cuda.empty_cache()

# Analyze results
task_scores = {task: [] for task in tasks}

for layer, result in results.items():
    for task in tasks:
        # Assuming the main metric is 'v_measure'. Adjust if needed.
        score = result[task]['v_measure']
        task_scores[task].append((layer, score))

# Rank layers for each task
for task, scores in task_scores.items():
    print(f"\nRanking for {task}:")
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for rank, (layer, score) in enumerate(sorted_scores, 1):
        print(f"Rank {rank}: Layer {layer} (Score: {score:.4f})")

# Overall ranking (average across tasks)
overall_scores = []
for layer in range(num_layers):
    avg_score = np.mean([task_scores[task][layer][1] for task in tasks])
    overall_scores.append((layer, avg_score))

print("\nOverall Ranking:")
sorted_overall = sorted(overall_scores, key=lambda x: x[1], reverse=True)
for rank, (layer, score) in enumerate(sorted_overall, 1):
    print(f"Rank {rank}: Layer {layer} (Average Score: {score:.4f})")