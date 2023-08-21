# source code for MatchXML, 2022

import torch
import torch.nn as nn

class DulConLoss(nn.Module):
    """Implementation is based on Supervised Contrastive Learning: 
    https://arxiv.org/pdf/2004.11362.pdf. """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(DulConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features_one, features_two, labels=None, mask=None):
        """Compute loss for model.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        size = features_one.shape[0]
        mask = (labels > 0).float()
        anchor_feature = features_one
        contrast_feature = features_two
        contrast_feature = torch.transpose(contrast_feature, -2, -1)
        #test = torch.matmul(anchor_feature, contrast_feature)
        #test = torch.squeeze(test,1)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.squeeze(torch.matmul(anchor_feature, contrast_feature),1),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -  mean_log_prob_pos
        loss = loss.view(1, size).mean()

        return loss
