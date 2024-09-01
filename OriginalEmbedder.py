#methods for original embedder used in distillation

import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OriginalEmbedder, self).__init__()
