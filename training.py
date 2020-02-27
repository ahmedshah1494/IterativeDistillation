import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import argparse
import logging
import sys
import models
import utils
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
from distillation import simple_distillation
from multiprocessing import cpu_count
from distillation import StudentModelWrapper

model_dict = {
    'resnet18TinyImageNet': models.resnet18TinyImageNet,
    'resnet18': models.resnet18,
    'vgg16': models.vgg16,
    'vgg16CIFAR': models.vgg16CIFAR,
    'AlexNetCIFAR' : models.alexnetCIFAR,
    'AlexNet' : models.alexnet,
    'PapernotCIFAR10' : models.papernot,
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

def evaluate_on_batch(model, batch, metric=utils.compute_correct):
    model.eval()
    x,y = batch
    if use_cuda:
        x = x.cuda()
        y = y.cuda()        
    out = model(x)
    return metric(out,y)

def evaluate(model, loader, label_counts=None):
    val_correct = 0
    if label_counts is not None:
        val_correct = np.zeros(len(label_counts))
        metric = lambda out, true: utils.compute_correct_per_class(out, true, len(label_counts))
    else:
        metric = utils.compute_correct
        
    for batch in loader:
        val_correct += evaluate_on_batch(model, batch, metric)
    
    return val_correct

def get_trainable_params(model,):
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            
    return params_to_update

def train(model, train_dataset, test_dataset, nclasses, args, val_dataset=None):    
    if val_dataset is None:
        new_train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - new_train_size
        train_dataset, val_dataset = random_split(train_dataset, [new_train_size, val_size])

    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=(cpu_count())//2)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=(cpu_count())//2) 
    if not args.test_only:
        criterion = utils.loss_wrapper(args.C)
        optimizer = torch.optim.SGD(get_trainable_params(model), lr=args.lr, weight_decay=5e-3, momentum=0.9)    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=0.5)

        val_label_counts = utils.label_counts(val_loader, nclasses)

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
            
            val_correct = evaluate(model, val_loader, val_label_counts)
            val_acc = np.mean(val_correct / val_label_counts)
            print('val_accuracy:', val_acc)

            if i == 0 or scheduler.is_better(val_acc, scheduler.best):            
                with open(args.outfile, 'wb') as f:
                    torch.save(model,f)

            scheduler.step(val_acc)

            logger.info('epoch#%d train_loss=%.3f train_acc=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_acc, val_acc, optimizer.param_groups[0]['lr']))
    
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=(cpu_count())//2)

    test_label_counts = utils.label_counts(test_loader, nclasses)    
    test_correct = evaluate(model, test_loader, test_label_counts)
    test_acc = np.mean(test_correct / test_label_counts)
    print('test_accuracy:', test_acc)
    logger.info('test_accuracy = %0.3f' % test_acc)


def main(args):
    train_dataset, test_dataset, nclasses = utils.get_datasets(args)

    if args.test_pc < 1:
        new_test_size = int(args.test_pc * len(test_dataset))
        test_dataset, _ = random_split(test_dataset, [new_test_size, len(test_dataset)-new_test_size])

    new_train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - new_train_size
    train_dataset, val_dataset = random_split(train_dataset, [new_train_size, val_size])

    if args.model_path is not None:
        model = torch.load(args.model_path)        
        if not args.feature_extraction:
            for param in model.parameters():
                param.requires_grad = True
        else:
            models.setup_feature_extraction_model(model, args.model, nclasses, args.classifier_depth)
    else:
        model = model_dict[args.model](nclasses, args.pretrained, args.feature_extraction, args.classifier_depth)

    if args.reinitialize:
        utils.reinitialize_model(model)

    if use_cuda:        
        model = model.cuda()
    logger.info(model)

    if args.pretrain_classifier_epochs > 0:
        models.setup_feature_extraction_model(model, args.model, nclasses, args.classifier_depth)
        
        if use_cuda:        
            model = model.cuda()

        nepochs_ = args.nepochs
        args.nepochs = args.pretrain_classifier_epochs
        train(model, train_dataset, test_dataset, nclasses, args, val_dataset)
        args.nepochs = nepochs_

        for param in model.parameters():
            param.requires_grad = True

    train(model, train_dataset, test_dataset, nclasses, args, val_dataset)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datafolder', type=str,default="/home/mshah1/workhorse3")
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--pretrain_classifier_epochs', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--C', type=float, default=0)
    parser.add_argument('--outfile', type=str, default='models/model.pt')
    parser.add_argument('--logfile', type=str, default='logs/training.log')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--reinitialize', action='store_true')
    parser.add_argument('--feature_extraction', action='store_true')
    parser.add_argument('--expand_w_multiple_scales', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_pc', type=float, default=1.0)
    parser.add_argument('--classifier_depth', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(1494)
    torch.manual_seed(1494)
    torch.cuda.manual_seed_all(1494)

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
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    logger.info(args)
    main(args)