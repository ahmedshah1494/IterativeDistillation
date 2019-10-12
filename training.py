import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import argparse
import logging
import sys
import models
import utils
import numpy as np
from tqdm import tqdm
import os

model_dict = {
    'AlexNetCIFAR' : models.AlexNetCIFAR,
    'AlexNet' : models.AlexNet,
    'PapernotCIFAR10' : models.PapernotCIFAR10,
    'PapernotCIFAR10_2' : models.PapernotCIFAR10_2,
}

def train_on_batch(model, batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    x,y = batch
    if use_cuda:
        x = x.cuda()
        y = y.cuda()        
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return float(loss*x.shape[0]), float(utils.compute_correct(out, y))

def evaluate_on_batch(model, batch):
    model.eval()
    x,y = batch
    if use_cuda:
        x = x.cuda()
        y = y.cuda()        
    out = model(x)
    return float(utils.compute_correct(out,y))

def loss(margin):
    if margin == 0:
        return nn.functional.cross_entropy
    else:
        return lambda x,y: nn.functional.multi_margin_loss(x/torch.sum(x,1,keepdim=True), y, margin=margin)

def train(model, train_dataset, val_dataset, args):
    train_loader = DataLoader(train_dataset, args.batch_size)
    val_loader = DataLoader(val_dataset, args.batch_size)

    patience = 5
    # criterion = nn.CrossEntropyLoss()
    criterion = loss(args.C)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=0.5)    
    
    for i in range(args.nepochs):
        epoch_loss = 0
        epoch_correct = 0

        t = tqdm(enumerate(train_loader))
        t.set_description('epoch#%d' % i)
        for j,batch in t:
            train_loss, train_correct = train_on_batch(model, batch, optimizer, criterion)
            epoch_loss += train_loss
            epoch_correct += train_correct
            t.set_postfix(loss=epoch_loss/((j+1)*args.batch_size), accuracy=epoch_correct/((j+1)*args.batch_size), 
                            lr=optimizer.param_groups[0]['lr'])                        
        epoch_loss /= len(train_dataset)
        epoch_acc = epoch_correct/len(train_dataset)
        
        val_correct = 0
        for batch in val_loader:
            val_correct += evaluate_on_batch(model, batch)
        val_acc = val_correct / len(val_dataset) 
        print('val_accuracy =',val_acc)
        
        if i == 0 or scheduler.is_better(val_acc, scheduler.best):            
            with open(args.outfile, 'wb') as f:
                torch.save(model,f)

        scheduler.step(val_acc)

        logger.info('epoch#%d train_loss=%.3f train_acc=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_acc, val_acc, optimizer.param_groups[0]['lr']))
def main(args):
    transform = torchvision.transforms.Compose([        
        torchvision.transforms.RandomAffine(30), 
        # torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        torchvision.transforms.RandomGrayscale(p=0.1),                
        ])
    transform = utils.curry(lambda x: torch.from_numpy(np.array(x)).float(), transform)
    transform = utils.curry(lambda x: x/255, transform)
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,1,-1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1,1,-1)
    transform = utils.curry(lambda x: (x - norm_mean) / norm_std, transform)
    if args.dataset == 'CIFAR10':
        transform = utils.curry(lambda x: x.transpose(2,1).transpose(0,1), transform)
        train_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', 
                    transform=transform, download=True)
        val_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', train=False,
                    transform=transform, download=True)        
        nclasses = 10
    if args.dataset == 'CIFAR100':
        transform = utils.curry(lambda x: x.transpose(2,1).transpose(0,1), transform)
        train_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', 
                    transform=transform, download=True)
        val_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', train=False,
                    transform=transform, download=True)        
        nclasses = 100
    elif args.dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageNet('/home/mshah1/workhorse3/', train=True,
                                                transform=transform, download=True)
        test_dataset = torchvision.datasets.ImageNet('/home/mshah1/workhorse3/', train=False,
                                                transform=transform, download=True)
        nclasses = 1000
    else:
        raise NotImplementedError
        
    if args.model_path is not None:
        model = torch.load(args.model_path)
    else:
        model = model_dict[args.model](nclasses)

    if use_cuda:        
        model = model.cuda()
    train(model, train_dataset, val_dataset, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--C', type=float, default=0)
    parser.add_argument('--outfile', type=str, default='model.pt')
    parser.add_argument('--logfile', type=str, default='training.log')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.logfile)):
        os.makedirs(os.path.dirname(args.logfile))
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(args.logfile),
        # logging.StreamHandler(sys.stdout)
    ])
    logger = logging.getLogger()

    use_cuda = args.cuda and torch.cuda.is_available()
    logger.info(args)
    main(args)