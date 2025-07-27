import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.5):
        super(TripletLoss,self).__init__()
        self.margin = margin

    
    def forward(self, anchor, positive,negative):

        # Расстояние междку векторами 
        rastoynie_pos = F.pairwise_distance(anchor,positive, p=2)
        rastoynie_neg = F.pairwise_distance(anchor,negative, p=2)

        #Tripelet loss
        loss = F.relu(rastoynie_pos - rastoynie_neg + self.margin)

        return loss.mean()

