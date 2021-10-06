import utils
import models
import solver

import argparse, os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np 
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(description='Step Training')

parser.add_argument('--gpu', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--in-dataset', default="Cifar10", type=str, help='ID dataset')
parser.add_argument('--out-dataset', default="LSUN", type=str, help='OOD dataset')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,  metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--epochs', default=1500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
parser.add_argument('--labels', dest='labels', type=int, default=250, help='n_labels')

def getProjectFeatures(projector, dataloader):
    features_arr, labels_arr = [], []
    projector.eval()
    for images, labels in dataloader:
        features = projector(images.cuda()).detach().cpu()
        features_arr.append(features)
        labels_arr.append(labels)
    features = torch.cat(features_arr, 0)
    labels = torch.cat(labels_arr, 0)
    return features.numpy(), labels.numpy()

def main(): 
    args = parser.parse_args()
    # Set Random Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Enable Cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    # Load data
    args.num_classes = 10 if args.in_dataset == "Cifar10" else 100
    Tr_L, Tr_U, Va, Te = utils.load_data(in_dataset=args.in_dataset, out_dataset=args.out_dataset, seed=args.seed, n_labels=args.labels)

    # Init Model
    backbone = models.DenseNet().cuda()
    projector = nn.Linear(backbone.n_features, backbone.n_features).cuda()
    classifier = models.Classifier(backbone.n_features, args.num_classes).cuda()
    task_name = f"{args.in_dataset}-{args.out_dataset}"
    checkpoint_name = f'OOD-{task_name}-{args.seed}.pth.tar'
    model = solver.OODSolver(backbone=backbone, fc=classifier, args=args)
    
    print(">>> Preparing Features")
    
    # Build dataset
    all_data = np.vstack([Tr_L["images"], Tr_U["images"]])
    all_data = model.getLayerFeatures(all_data)
    Tr_L_data = model.getLayerFeatures(Tr_L["images"])
    Va_data = model.getLayerFeatures(Va["images"])
    Te_data = model.getLayerFeatures(Te["images"])

    triple_dataset = model.build(X=Tr_L_data, all_data=all_data)
    train_dataset = utils.RawDataset(X=Tr_L_data, y=Tr_L['labels'])
    triple_loader = torch.utils.data.DataLoader(triple_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    
    print(">>> Start Training Projector")

    # Train projector
    model.train(triple_loader, n_features=all_data.shape[1])

    print(">>> Start Testing")

    # Load model
    checkpoint = torch.load(os.path.join("files", checkpoint_name))
    projector = nn.Linear(checkpoint['n_features'], backbone.n_features).cuda()
    backbone.load_state_dict(checkpoint['backbone'])
    projector.load_state_dict(checkpoint['projector'])
    backbone.eval()
    projector.eval()

    # Prepare testing data
    Tr_L_data = model.getLayerFeatures(Tr_L["images"])
    Te_data = model.getLayerFeatures(Te["images"])
    label_dataset = utils.RawDataset(X=Tr_L_data, y=Tr_L['labels'])   
    label_loader = torch.utils.data.DataLoader(label_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_dataset = utils.RawDataset(X=Te_data, y=Te['labels'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    Tr_features, Tr_labels = getProjectFeatures(projector, label_loader)
    Te_features, Te_labels = getProjectFeatures(projector, test_loader)

    # Calculate class center
    centers = []
    for i in range(args.num_classes):
        index = Tr_labels == i
        center = np.mean(Tr_features[index], axis=0)
        centers.append(center)
    
    # Calculate OOD score
    cifar, other = [], []
    for (f, l) in tqdm(zip(Te_features, Te_labels)):
        score = 1e18
        for i in range(args.num_classes):
            score = min(score, np.mean((f - centers[i]) ** 2))
        if l >= 0:
            cifar.append(score)
        else:
            other.append(score)
    cifar, other = np.array(cifar), np.array(other)

    # Evaluate by metrics
    print(">>> Results")
    auroc = utils.auroc(cifar, other)
    tpr95 = utils.tpr95(cifar, other)
    auprIn = utils.auprIn(cifar, other)
    auprOut = utils.auprOut(cifar, other)
    detection = utils.detection(cifar, other)
    utils.save_json(f"OOD-{task_name}", args.seed, {
        "auroc": auroc,
        "tpr95": tpr95,
        "auprIn": auprIn,
        "auprOut": auprOut,
        "detection": detection,
    })

if __name__ == "__main__": main()