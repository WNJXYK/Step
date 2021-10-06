import utils
import models
import solver
from torchvision import transforms
import argparse, os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np 

parser = argparse.ArgumentParser(description='Unsupervised Pretrain')

parser.add_argument('--wide', dest='wide', action='store_true', help='use wide-resnet')
parser.add_argument('--in-dataset', default="Cifar10", type=str, help='ID dataset')
parser.add_argument('--out-dataset', default="LSUN", type=str, help='OOD dataset')
parser.add_argument('--gpu', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers (default: 12)')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,  metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.')
parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
parser.add_argument('--out-dim', default=256, type=int, help='feature dimension (default: 128)')

        
def main():
    args = parser.parse_args()
    
    # Fix Random for pretrained model
    args.seed = 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Enable Cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    # Load data
    n_labels = (10 if args.in_dataset == 'Cifar10' else 100)
    Tr_L, Tr_U, Va, Te = utils.load_data(in_dataset=args.in_dataset, out_dataset=args.out_dataset, seed=args.seed, n_labels=n_labels)
    X = np.vstack([Tr_L["images"], Tr_U["images"]])
    print("Unsupervised Train Set", X.shape)
    train_dataset = utils.ContrastiveDataset(X=X)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # Init Model
    backbone = models.WideResNet().cuda() if args.wide else models.DenseNet().cuda()
    print("Backbone Feature Dim:", backbone.n_features)
    print("Unsupervised Feature Dim:", args.out_dim)
    classifier = models.Classifier(backbone.n_features, args.out_dim).cuda()
    model = models.SimCLR(backbone=backbone, fc=classifier).cuda()
    if torch.cuda.device_count() > 1:
        print("Training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Unsupervised Trainning
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    simclr = solver.SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)

if __name__ == "__main__": main()