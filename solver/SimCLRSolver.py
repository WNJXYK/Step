import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import save_config_file, accuracy, save_checkpoint

__all__ = ['SimCLR']

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        torch.manual_seed(self.args.seed)
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.name = f"{self.args.in_dataset}-{self.args.out_dataset}"
        if self.args.wide: self.name += '-W'
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        for epoch_counter in trange(self.args.epochs):
            for images in train_loader:
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            if epoch_counter >= 10: self.scheduler.step()

        # Save Model
        checkpoint_name = f'SimCLR-{self.name}.pth.tar'
        save_checkpoint({
            'args': self.args,
            'backbone': self.model.backbone.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join("files", checkpoint_name))