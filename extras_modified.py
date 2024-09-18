import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive samples
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        
        
        positives = similarity_matrix.masked_select(mask).view(similarity_matrix.size(0), 1)
        negatives = similarity_matrix[~mask].view(similarity_matrix.size(0), -1) # using the negation (~) of the identity mask directly to select negative samples
        
        # Concatenate the positive samples back to their corresponding positions
        logits = torch.cat((positives, negatives), dim=1)

        # Compute labels: positive sample is always the first one
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=similarity_matrix.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]