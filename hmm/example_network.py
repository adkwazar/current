import torch
import torch.nn as nn
import torch.nn.functional as F

class AAModel(nn.Module):

    def __init__(self, context_size, HD):
        super().__init__()
        self.embeddings = nn.Embedding(20, 2) #20 amino acids, each of them represented as 2D point (vector)
        self.linear1 = nn.Linear(context_size*2, HD) #context_size = N-1 (notation from paper), HD = number of neurons in a hidden layer
        self.linear2 = nn.Linear(HD, 20) 

    def forward(self, inputs):
        representations = self.embeddings(inputs).view((1, -1)) 
        out1 = F.relu(self.linear1(representations)) 
        out2 = self.linear2(out1)
        log_probs = F.log_softmax(out2, dim=1) 
        return log_probs