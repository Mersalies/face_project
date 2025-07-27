import torch.nn as nn
import torch.nn.functional as F


class Face_model(nn.Module):

    def __init__(self, embedding_size = 128):
        super(Face_model, self).__init__()

        self.convnet = nn.Sequential(

            nn.Conv2d(3,64,7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))

        )
        self.embedding = nn.Linear(256, embedding_size)


    def forward(self,x):
        
        x = self.convnet(x)
        x = x.view(x.size(0),-1)
        x = self.embedding(x)

        return F.normalize(x,p=2,dim=1)





