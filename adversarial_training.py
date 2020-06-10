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
from distillation import StudentModelWrapper, StudentModelWrapper2

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, FGSM, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ChooseBestAttack
from advertorch.attacks.utils import attack_whole_dataset
from advertorch_examples.utils import get_cifar10_test_loader, get_test_loader
from advertorch.utils import get_accuracy
from advertorch.loss import CWLoss

from attack_classifier import NormalizationWrapper, extract_attack

model_dict = {
    'LeNet': models.lenet,
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
    device = [p for p in model.parameters()][0].device
    optimizer.zero_grad()
    x,y = batch
    # if use_cuda:
    x = x.to(device)
    y = y.to(device)        
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return float(loss*x.shape[0]), float(utils.compute_correct(out, y))

def evaluate_on_batch(model, batch, metric=utils.compute_correct):
    model.eval()
    device = [p for p in model.parameters()][0].device
    x,y = batch
    # if use_cuda:
    x = x.to(device)
    y = y.to(device)        
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

def train(model, train_dataset, test_dataset, nclasses, adversary, args, val_dataset=None, mLogger=None):
    print(mLogger)
    if mLogger is not None:
        logger = mLogger
    if val_dataset is None:
        new_train_size = int(0.8*len(train_dataset))
        val_size = len(train_dataset) - new_train_size
        train_dataset, val_dataset = random_split(train_dataset, [new_train_size, val_size])

    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=(cpu_count())//2)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=(cpu_count())//2) 
    
    criterion = utils.loss_wrapper(args.C)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(get_trainable_params(model), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(get_trainable_params(model), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=0.2)

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=(cpu_count())//2)    
    test_label_counts = utils.label_counts(test_loader, nclasses)    
    test_correct = evaluate(model, test_loader, test_label_counts)
    test_acc = np.sum(test_correct) / np.sum(test_label_counts)
    print('test_accuracy:', test_acc)
    logger.info('test_accuracy = %0.3f' % test_acc)

    val_label_counts = utils.label_counts(val_loader, nclasses)
    bad_iters = 0
    for i in range(args.nepochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        t = tqdm(enumerate(train_loader))
        t.set_description('epoch#%d' % i)
        for j,batch in t:
            x, y = batch
            x = x.cuda()
            y = y.cuda()

            if args.gaussian_smoothing:
                eps = torch.normal(mean=0, std=args.sigma,size=x.shape).cuda()
                x += eps
            else:
                flips = np.random.binomial(1, 0.5, size=x.shape[0])
                flips = flips == 1
                x[flips] = adversary.perturb(x[flips],y[flips])
            
            train_loss, train_correct = train_on_batch(model, (x,y), optimizer, criterion)
            epoch_loss += train_loss
            epoch_correct += train_correct
            epoch_count += x.shape[0]
            t.set_postfix(loss=epoch_loss/((j+1)*args.batch_size), accuracy=epoch_correct/(epoch_count), 
                            lr=optimizer.param_groups[0]['lr'])
        epoch_loss /= len(train_dataset)
        epoch_acc = epoch_correct/len(train_dataset)
        
        # val_correct = evaluate(model, val_loader, val_label_counts)
        # val_acc = np.mean(val_correct / val_label_counts)
        # print('val_accuracy:', val_acc, )

        adv, label, pred, advpred = attack_whole_dataset(adversary, val_loader)
        val_acc = get_accuracy(pred, label)
        adv_acc = get_accuracy(advpred, label)
        print('clean val accuracy:', val_acc)
        print('robust val accuracy:', adv_acc)

        if i == 0 or scheduler.is_better(val_acc, scheduler.best):            
            with open(args.outfile, 'wb') as f:
                torch.save(model,f)
            bad_iters = 0
        else:
            bad_iters += 1
        if bad_iters >= 3 * args.patience:
            print('early stopping...')
            break
        scheduler.step(adv_acc)

        logger.info('epoch#%d train_loss=%.3f train_acc=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_acc, val_acc, optimizer.param_groups[0]['lr']))
    
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=(cpu_count())//2)
    model = torch.load(args.outfile)
    test_label_counts = utils.label_counts(test_loader, nclasses)    
    test_correct = evaluate(model, test_loader, test_label_counts)
    test_acc = np.sum(test_correct) / np.sum(test_label_counts)
    print('test_accuracy:', test_acc)
    logger.info('test_accuracy = %0.3f' % test_acc)

    adv, label, pred, advpred = attack_whole_dataset(adversary, test_loader)
    test_acc = get_accuracy(pred, label)
    print('clean test accuracy:', test_acc)
    print('robust test accuracy:',get_accuracy(advpred, label))


def main(args):
    train_dataset, test_dataset, nclasses = utils.get_datasets(args, normalize=False)

    new_train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - new_train_size
    train_dataset, val_dataset = random_split(train_dataset, [new_train_size, val_size])

    if args.model_path is not None:
        if args.dataset == 'cifar10': 
            model = (torch.load(args.model_path))
        else:
            model = torch.load(args.model_path)
    else:
        model = model_dict[args.model](nclasses)
    
    if args.reinitialize:
        utils.reinitialize_model(model)

    if use_cuda:        
        model = model.cuda()
    logger.info(model)

    attack_class,attack_kwargs = extract_attack(args)    
    attackers = [attack_class(model, **attack_kwargs) for i in range(args.nb_restarts)]
    if len(attackers) > 1:
        attacker = ChooseBestAttack(model, attackers, targeted=attackers[0].targeted)
    else:
        attacker = attackers[0]

    train(model, train_dataset, test_dataset, nclasses, attacker, args, val_dataset, mLogger=logger)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datafolder', type=str,default="/home/mshah1/workhorse3")
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam','sgd'))
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--C', type=float, default=0)
    parser.add_argument('--outfile', type=str, default='models/model.pt')
    parser.add_argument('--logfile', type=str, default='logs/training.log')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--reinitialize', action='store_true')
    parser.add_argument('--attack', type=str, default="none", choices=('none', 'pgdinf', 'pgdl2'))
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=7)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--eps_iter', type=float, default=8/(255*4))
    parser.add_argument('--max_iters', type=float, default=7)
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--gaussian_smoothing', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.25)
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