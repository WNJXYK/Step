import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import save_config_file, accuracy, save_checkpoint, AverageMeter
import utils
import numpy as np
import faiss, time
from scipy.linalg import sqrtm
import copy
__all__ = ['OODSolver']

class OODSolver(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        torch.manual_seed(self.args.seed)
        self.name = f"{self.args.in_dataset}-{self.args.out_dataset}"
        self.backbone = kwargs['backbone'].to(self.args.device)
        
        checkpoint_name = f'SimCLR-{self.name}.pth.tar'
        resume = os.path.join("./files", checkpoint_name)
        checkpoint = torch.load(resume)
        self.backbone.load_state_dict(checkpoint['backbone'])
        print("Loaded checkpoint '{}'".format(resume))
    
        self.projector = None
        self.optimizer = None
        self.n_features = 0

    def getLayerFeatures(self, X, layerLevel=[0, 1, 2, 3]):
        dataset = utils.UDataset(X=X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
        features_arr = []
        self.set_eval()
        for images in tqdm(loader):
            images = images.to(self.args.device)
            cur = []
            for level in layerLevel:
                featuresH = self.backbone(images, level=level)
                cur.append(featuresH.detach().cpu())
            features_arr.append(torch.cat(cur, 1))
        features = torch.cat(features_arr, 0).numpy()
        print("Feature Space", features.shape)
        return features

    def build(self, X, all_data, K=12):
        cov = np.cov(X.T)
        L = np.linalg.cholesky(cov + np.eye(cov.shape[0]))
        mahalanobis_transform = np.linalg.inv(L)
        features = np.dot(all_data, mahalanobis_transform.T).astype('float32')
        n_samples = features.shape[0]
        n_features = features.shape[1]
        graph = faiss.IndexFlatL2(n_features)
        graph.add(features)
        D, I = graph.search(features, K)
        return utils.TripleDataset(X=all_data, I=I, D=D)

    def set_train(self):
        self.backbone.train()
        if self.projector is not None:
            self.projector.train()
    
    def set_eval(self):
        self.backbone.eval()
        if self.projector is not None:
            self.projector.eval()

    def structure_loss(self, fO, fK, fP, fD, margin=3):
        fDist = F.mse_loss(fO, fK, reduction="none").sum(1)
        fDist = torch.clamp(fDist - fD ** 2, min=0).mean()
        rDist = F.mse_loss(fO, fP, reduction="mean")
        rDist = torch.clamp(margin - rDist, min=0)
        return fDist + rDist
    
    def structure_samples(self):
        try:
            fO, fK, fP, fD = self.triple_iter.next()
        except StopIteration:
            self.triple_iter = iter(self.triple_loader)
            fO, fK, fP, fD = self.triple_iter.next()
        return fO, fK, fP, fD

    def train(self, triple_loader, n_features):
        checkpoint_name = f'OOD-{self.name}-{self.args.seed}.pth.tar'
        self.triple_loader = triple_loader
        self.triple_iter = iter(self.triple_loader)

        self.n_features = n_features
        self.projector = nn.Linear(n_features, self.backbone.n_features).to(self.args.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.projector.parameters()},
        ], self.args.lr, weight_decay=self.args.weight_decay)
        
        tr_loss = AverageMeter()
        message = trange(self.args.epochs)
        for epoch_counter in message:
            self.set_train()
            fO, fK, fP, fD = self.structure_samples()
            fO, fK, fP, fD = fO.to(self.args.device), fK.to(self.args.device), fP.to(self.args.device), fD.to(self.args.device)
            fO, fK, fP = self.projector(fO), self.projector(fK), self.projector(fP)
            loss = self.structure_loss(fO, fK, fP, fD)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tr_loss.update(loss.item(), fO.shape[0])
            message.set_description(f"Loss = {tr_loss.avg}")
           
            save_checkpoint({
                'args': self.args,
                'backbone': self.backbone.state_dict(),
                'projector': self.projector.state_dict(),
                "n_features": self.n_features,
            }, is_best=False, filename=os.path.join("files", checkpoint_name))
