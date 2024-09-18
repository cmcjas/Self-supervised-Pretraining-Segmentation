import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # Normalize feature vectors
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Mask to select positive examples (diagonal)
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        positives = similarity_matrix.masked_select(mask)

        # Mask to zero out positives, leaving only negatives
        # negatives_mask = ~mask

        # For each positive, calculate the log-sum-exp of all negatives
        negatives = similarity_matrix.masked_fill(mask, float('-inf'))  # Fill diagonals (positives) with -inf to ignore them
        negatives_logsumexp = torch.logsumexp(negatives, dim=1)

        # print(f"Positives shape: {positives.shape}")
        # print(f"Negatives shape: {negatives.shape}")

        # Compute InfoNCE loss
        loss = -positives + negatives_logsumexp
        return loss.mean()

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]