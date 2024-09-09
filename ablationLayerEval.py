#this script uses a subset of clustering evals to decide what layers to ablate
#it is a proper subset to minimize time and compute, as MTEB evals tend to have a lot of data

from datasets import load_dataset
import torch
from ablationEstimator import AblationLayerEstimator
import numpy as np
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans
import random

# Load datasets
ds1 = load_dataset("mteb/arxiv-clustering-p2p")
ds2 = load_dataset("mteb/arxiv-clustering-s2s")

# List of tasks and their corresponding datasets
tasks = {
    "ArxivClusteringP2P": ds1["test"],
    "ArxivClusteringS2S": ds2["test"]
}

num_layers = 32
results = {}

def evaluate_clustering(embeddings, labels):
    # Perform K-means clustering
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    predicted_clusters = kmeans.fit_predict(embeddings)
    
    # Calculate V-measure score
    v_measure = v_measure_score(labels, predicted_clusters)
    return v_measure



for layer in range(num_layers):
    print(f"Evaluating model with layer {layer} ablated...")
    model = AblationLayerEstimator(layer).to('cuda')
    
    task_results = {}
    for task_name, dataset in tasks.items():
        # Get a subset of the dataset
        
        subset = dataset[0] #dataset is 31,2, but in each row of 31, there is a list of sentences that match with a list of labels
        #for this basic eval for ablation testing, we only really need the first list matches
        
        
        # Get embeddings
        embeddings = model.encode(subset['sentences'],layer=layer)  # Changed 'text' to 'sentences'
        
        # Evaluate clustering
        v_measure = evaluate_clustering(embeddings.cpu().numpy(), subset['labels'])  # Changed 'label' to 'labels'
        
        task_results[task_name] = {'v_measure': v_measure}
    
    # Store results
    results[layer] = task_results

    # Clear CUDA cache after each layer evaluation
    torch.cuda.empty_cache()


# Analyze results
task_scores = {task: [] for task in tasks}

for layer, result in results.items():
    for task in tasks:
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