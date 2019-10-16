import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import argparse
import logging
import sys
import models
import utils
import numpy as np
from tqdm import tqdm,trange
from multiprocessing import cpu_count
from copy import deepcopy
import os

class StudentModelWrapper(nn.Module, models.ModelWrapper):
    def __init__(self, model):
        super(StudentModelWrapper, self).__init__()        
        containers = [nn.Sequential, nn.ModuleList, nn.ModuleDict, type(model)]
        is_container = lambda m: 1 in [isinstance(m,t) for t in containers]
        layers = [m for m in model.modules() if not is_container(m)]
        for i,l in enumerate(layers):
            if isinstance(l, nn.AdaptiveAvgPool2d) and i < len(layers)-1 and not isinstance(layers[i+1], models.Flatten):
                layers.insert(i+1, models.Flatten())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def cross_entropy(logits_p, logits_q, T):
    p = F.softmax(logits_p/T, dim=1)
    q = F.log_softmax(logits_q/T, dim=1)
    return torch.mean(torch.sum(-p*q, dim=1))

def distillation_loss(student_out, teacher_out, labels, loss_weight=1e-4, C=1, T=1):
    # xent = F.cross_entropy(student_out, labels)    
    hard_loss = utils.loss_wrapper(C)(student_out, labels)
    # soft_loss = F.kl_div(F.log_softmax(student_out/T, dim=1), F.softmax(teacher_out/T, dim=1), reduction='batchmean')
    soft_loss = cross_entropy(teacher_out, student_out, T)
    return soft_loss*(1-loss_weight)*T**2 + loss_weight*hard_loss, soft_loss, hard_loss

def copy_model(model, device, reinitialize=False):
    new_model = deepcopy(model)
    if reinitialize:
        for param in new_model.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
            else:
                torch.nn.init.constant_(param.data,0)
    return new_model

def evaluate(model, val_dataset, args):
    val_loader = DataLoader(val_dataset, 1024, shuffle=False)
    val_correct = 0
    for batch in val_loader:
        model.eval()
        x,y = batch        
        x = x.to(args.device)
        y = y.to(args.device)
        out = model(x)
        val_correct += float(utils.compute_correct(out,y))
    val_acc = val_correct / len(val_dataset) 
    return val_acc

def distill(student_model, teacher_model, train_dataset, val_dataset, base_metric, args):
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=False)

    criterion = distillation_loss
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise ValueError('optimizer should be either "adam" or "sgd"')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=0.5)

    best_model = None
    t = trange(args.nepochs)
    for i in t:
        epoch_loss = 0
        epoch_soft_loss = 0
        epoch_hard_loss = 0
        epoch_correct = 0
        count = 0
        # t = tqdm(enumerate(train_loader))
        # t.set_description('epoch#%d' % i)
        for j,batch in enumerate(train_loader):
            student_model = student_model.train()
            optimizer.zero_grad()
            x,y = batch
            x,y = x.to(args.device), y.to(args.device)
            z = teacher_model(x)
            z_ = student_model(x)
            train_loss, train_soft_loss, train_hard_loss = criterion(z_, z, y, loss_weight=args.loss_weight, C=args.C, T=args.T)
            train_loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(train_loss)
            epoch_soft_loss += float(train_soft_loss)
            epoch_hard_loss += float(train_hard_loss)
            epoch_correct += float(utils.compute_correct(z_, y))
            count += 1
            # t.set_postfix(loss=epoch_loss/((j+1)*args.batch_size), lr=optimizer.param_groups[0]['lr'], accuracy=epoch_correct/((j+1)*args.batch_size)) 
        epoch_loss /= count
        epoch_soft_loss /= count
        epoch_hard_loss /= count

        val_correct = 0
        val_loss = 0
        count = 0
        for batch in val_loader:
            student_model = student_model.eval()
            x,y = batch
            x,y = x.to(args.device), y.to(args.device)
            z = teacher_model(x)
            z_ = student_model(x)
            val_loss += float(criterion(z_, z, y, loss_weight=args.loss_weight, C=args.C, T=args.T)[0])
            val_correct += float(utils.compute_correct(z_,y))
            count += 1
        val_loss /= count          
        val_acc = val_correct / len(val_dataset)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        # if optimizer.param_groups[0]['lr'] != old_lr:
        #     args.T = min(args.max_T, args.T*2)
        t.set_postfix(train_loss=epoch_loss, train_soft_loss=epoch_soft_loss, train_hard_loss=epoch_hard_loss, 
                        val_loss=val_loss, val_acc=val_acc, lr=optimizer.param_groups[0]['lr'], T=args.T)        
        if i % 10 == 0:
            logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
        if val_acc-base_metric >= args.tol:
            logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
            break
    return scheduler.best
def iterative_distillation(teacher_model, student_model:StudentModelWrapper, bottleneck_layer_idx, train_dataset, val_dataset, test_dataset, nclasses, args):
    print('shrinking layer %d...' % bottleneck_layer_idx)    
    logger.info('shrinking layer %d...' % bottleneck_layer_idx)    
    student_model.shrink_layer(bottleneck_layer_idx, factor=args.shrink_factor)    
    student_model = student_model.to(args.device)     
    bottleneck_size = student_model.layers[bottleneck_layer_idx].weight.shape[0]
    delta = 0
    base_metric = evaluate(teacher_model, val_dataset, args)
    niters = 0
    final = False
    print('base_metric = %.4f' % (base_metric))
    logger.info('base_metric = %.4f' % (base_metric))
    while (bottleneck_size > 1 or final):  
        print('bottleneck_size = %d' % bottleneck_size)
        logger.info('bottleneck_size = %d' % bottleneck_size)        
        metric = distill(student_model, teacher_model, train_dataset, val_dataset, base_metric, args)
        logger.info('current metric = %.4f base_metric = %.4f' % (metric, base_metric))

        delta = metric - base_metric
        if delta < args.tol:
            logger.info('tolerance violated; stopping distillation')
            break
        else:        
            logger.info('saving model...')
            torch.save(student_model, args.outfile)
        
        student_model.shrink_layer(bottleneck_layer_idx, factor=args.shrink_factor)    
        student_model = student_model.to(args.device)     
        bottleneck_size = student_model.layers[bottleneck_layer_idx].weight.shape[0]
        if final:
            break
        if bottleneck_size == 1 and not final:
            final = True
    evaluate(teacher_model, test_dataset, args)
    print('test_metric = %.4f' % (base_metric))
    logger.info('test_metric = %.4f' % (base_metric))
    return student_model

def simple_distillation(teacher_model: models.ModelWrapper, student_model, train_dataset, val_dataset, nclasses, args):
    base_metric = evaluate(teacher_model, val_dataset, args)
    print('base_metric = %.4f' % (base_metric))
    logger.info('base_metric = %.4f' % (base_metric))
    distill(student_model, teacher_model, train_dataset, val_dataset, base_metric, args)    

def main(args):
    train_dataset, test_dataset, nclasses = utils.get_datasets(args)
    new_test_size = int(0.8*len(test_dataset))
    val_size = len(test_dataset) - new_test_size
    test_dataset, val_dataset = random_split(test_dataset, [new_test_size, val_size])
    
    teacher_model = torch.load(args.teacher_model_file).to(args.device) 
    teacher_model = teacher_model.eval()

    if args.student_model_file is None:
        student_model = copy_model(teacher_model, args.device, reinitialize=(not args.retain_teacher_weights))
        student_model = StudentModelWrapper(student_model)
        for param in student_model.parameters():
            param.requires_grad = True
    else:
        student_model = torch.load(args.student_model_file)
    shrinkable_layers = student_model.get_shrinkable_layers()
    iterative_distillation(teacher_model, student_model, shrinkable_layers[-1], train_dataset, val_dataset, test_dataset, nclasses, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--teacher_model_file', type=str)
    parser.add_argument('--student_model_file', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam/sgd')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--C', type=float, default=1)
    parser.add_argument('--T', type=float, default=1)
    parser.add_argument('--loss_weight', type=float, default=1e-4)
    parser.add_argument('--max_T', type=float, default=100)
    parser.add_argument('--tol', type=float, default=-0.05)
    parser.add_argument('--shrink_factor', type=float, default=0.9)
    parser.add_argument('--outfile', type=str, default='models/distillation/distilled_model.pt')
    parser.add_argument('--logfile', type=str, default='logs/distillation/distillation.log')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--retain_teacher_weights', action='store_true')
    args = parser.parse_args()

    np.random.seed(1494)
    torch.manual_seed(1494)
    
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

    logger.info(args)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)