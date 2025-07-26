import torch.nn as nn
import torch.nn.functional as F


class Face_model(nn.Module):

    def __init__(self, embedding_size = 128):
        super(Face_model, self).__init__()

        self.convnet = nn.Sequential(

            nn.Conv2d(3,64,7, stride=2, padding=3),


        )
