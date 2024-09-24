# DistilledEmbedding
A prototype set of scripts to quickly train distilled and pruned versions of Nvidia-Embedv2 (https://huggingface.co/nvidia/NV-Embed-v2). 

Distillation:
We ablate layers and evaluate performance on Arxiv clustering tasks (a small subset of MTEB, https://huggingface.co/spaces/mteb/leaderboard); the layers with least importance are removed, and the new, smaller model is fine-tuned with LoRA and with KL divergence to match the logit distribution of the teacher model on a text embedding dataset (https://huggingface.co/datasets/embedding-data/WikiAnswers).
See distillation.py

Pruning:
We evaluate the embedding logit activations on a small diverse and high quality text dataset (https://huggingface.co/datasets/nampdn-ai/tiny-textbooks), then prune the embedding neurons to reduce the embedding dimension, which comes with speed up advantages in practice associated with reduced embedding size (vector db, faster cosine similarity, etc.). We then fine-tune on the same dataset as the distilled model, but with similarity scores instead of student-teacher distribution matching. See prune.py


potentially will upload model weights in the future to huggingface
