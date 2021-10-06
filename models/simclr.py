import torch.nn as nn

__all__ = ['SimCLR']

class SimCLR(nn.Module):
    def __init__(self, backbone, fc):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        # Add a MLP Projection
        dim_mlp = self.backbone.n_features
        self.projector= nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU())
        self.fc = fc

    def forward(self, x):
        feature = self.backbone(x)
        projection = self.projector(feature)
        return self.fc(projection)