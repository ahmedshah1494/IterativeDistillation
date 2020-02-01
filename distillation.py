import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
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
    def __init__(self, model, logger):
        super(StudentModelWrapper, self).__init__()        
        self.logger = logger
        containers = [nn.Sequential, nn.ModuleList, nn.ModuleDict, type(model)]
        is_container = lambda m: 1 in [isinstance(m,t) for t in containers]
        layers = [m for m in model.modules() if not is_container(m)]
        for i,l in enumerate(layers):
            if isinstance(l, nn.AdaptiveAvgPool2d) and i < len(layers)-1 and not isinstance(layers[i+1], models.Flatten):
                layers.insert(i+1, models.Flatten())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class StudentModelWrapper2(nn.Module, models.ModelWrapper2):
    def __init__(self, model, logger, args):
        super(StudentModelWrapper2, self).__init__()
        self.logger = logger
        self.args = args
        containers = [nn.Sequential, nn.ModuleList, nn.ModuleDict, type(model)]
        is_container = lambda m: 1 in [isinstance(m,t) for t in containers]
        layers = [m for m in model.modules() if not is_container(m)]
        for i,l in enumerate(layers):
            if isinstance(l, nn.AdaptiveAvgPool2d) and i < len(layers)-1 and not isinstance(layers[i+1], models.Flatten):
                layers.insert(i+1, models.Flatten())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def to(self, device):
        self.device = device
        return super(StudentModelWrapper2, self).to(device)
        

def cross_entropy(logits_p, logits_q, T):
    p = F.softmax(logits_p/T, dim=1)
    q = F.log_softmax(logits_q/T, dim=1)
    return torch.mean(torch.sum(-p*q, dim=1))

def distillation_loss(student_out, teacher_out, labels, loss_weight=1e-4, C=1, T=1, soft_loss_type='xent'):
    if not torch.isfinite(student_out).any():
        print(student_out)
        exit(0)
    # xent = F.cross_entropy(student_out, labels)    
    hard_loss = utils.loss_wrapper(C)(student_out, labels)
    # soft_loss = F.kl_div(F.log_softmax(student_out/T, dim=1), F.softmax(teacher_out/T, dim=1), reduction='batchmean')
    if soft_loss_type == 'xent':
        soft_loss = cross_entropy(teacher_out, student_out, T)    
        return soft_loss*(1-loss_weight)*T**2 + loss_weight*hard_loss, soft_loss, hard_loss
    elif soft_loss_type == 'rmse':
        soft_loss = torch.sqrt(torch.mean((teacher_out - student_out)**2))
        return soft_loss*(1-loss_weight) + loss_weight*hard_loss, soft_loss, hard_loss
    else:
        raise NotImplementedError('soft_loss_type %s not supported' % soft_loss_type)

def copy_model(model, device, reinitialize=False):
    new_model = deepcopy(model)
    if reinitialize:
        for param in new_model.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
            else:
                torch.nn.init.constant_(param.data,0)
    return new_model

def evaluate(model, val_dataset, args, metric=utils.compute_correct):
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    val_correct = 0
    for batch in val_loader:
        model.eval()
        x,y = batch        
        x = x.to(args.device)
        y = y.to(args.device)
        out = model(x)
        val_correct += float(metric(out,y))
    val_acc = val_correct / len(val_dataset) 
    return val_acc

def get_logits(model, dataset, args):
    loader = DataLoader(dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=False)

    X = []
    Y = []
    Z = []
    for batch in loader:
        x,y = batch[:2]
        x,y = x.to(args.device), y.to(args.device)
        z = model(x)
        
        z = z.detach().cpu()
        x = x.detach().cpu()
        y = y.detach().cpu()

        X.append(x)
        Y.append(y)
        Z.append(z)
    
    X = torch.cat(X, 0)
    Y = torch.cat(Y, 0)
    Z = torch.cat(Z, 0)
    print(X.shape, Y.shape, Z.shape)
    logits = torch.utils.data.TensorDataset(X,Y,Z)
    return logits

def distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn=utils.compute_correct):    
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
    T = args.T
    loss_w = args.loss_weight
    for i in t:
        epoch_loss = 0
        epoch_soft_loss = 0
        epoch_hard_loss = 0
        epoch_correct = 0
        count = 0        
        for j,(x,y,z) in enumerate(train_loader):
            student_model = student_model.train()
            optimizer.zero_grad()
            x = x.to(args.device)
            z_ = student_model(x)
            y, z = y.to(args.device), z.to(args.device)
            train_loss, train_soft_loss, train_hard_loss = criterion(z_, z, y, 
                                                                    loss_weight=loss_w, 
                                                                    C=args.C, T=T, 
                                                                    soft_loss_type=args.soft_loss_type)
            train_loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(train_loss)
            epoch_soft_loss += float(train_soft_loss)
            epoch_hard_loss += float(train_hard_loss)
            epoch_correct += float(utils.compute_correct(z_, y))
            count += 1
        epoch_loss /= count
        epoch_soft_loss /= count
        epoch_hard_loss /= count

        val_correct = 0
        count = 0        
        for batch in val_loader:
            student_model = student_model.eval()
            x,y = batch
            x = x.to(args.device)
            z_ = student_model(x).detach().cpu()
            val_correct += float(val_metric_fn(z_,y))
            count += x.shape[0]         
        val_acc = val_correct / count

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        if optimizer.param_groups[0]['lr'] <= 1e-6:
            break
        
        if args.increase_loss_weight:
            loss_w += (1-args.loss_weight) / args.nepochs
        if args.anneal_temperature:
            T -= (args.T-1) / args.nepochs

        t.set_postfix(train_loss=epoch_loss, train_soft_loss=epoch_soft_loss, train_hard_loss=epoch_hard_loss, 
                        val_acc=val_acc, lr=optimizer.param_groups[0]['lr'], T=T, lossW=loss_w)        
        if i % 10 == 0:
            if 'logger' in locals():
                logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
        if val_acc-base_metric >= args.tol:
            if 'logger' in locals():
                logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4f' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
            break
    return scheduler.best
    
def iterative_distillation(student_model:StudentModelWrapper, bottleneck_layer_idx, train_dataset, val_dataset, test_dataset, nclasses, base_metric, args, mLogger=None, val_metric_fn=utils.compute_correct): 
    if mLogger is not None:
        logger = mLogger
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=False)

    def fine_tune():
        loss_w = args.loss_weight            
        args.loss_weight = 1

        nepochs = args.nepochs
        args.nepochs = args.n_fine_tune_epochs        

        metric = distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn)

        args.loss_weight = loss_w
        args.nepochs = nepochs
        
        torch.cuda.ipc_collect()
        return metric

    print(student_model) 
    
    torch.save(student_model, args.outfile)
    student_model = student_model.to(args.device)

    print('shrinking layer %d...' % bottleneck_layer_idx)    
    logger.info('shrinking layer %d...' % bottleneck_layer_idx)

    gpu_mem = torch.cuda.memory_allocated(args.device)
    
    bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
    
    batch_idx = np.random.permutation(np.arange(len(train_dataset)))[:max(bottleneck_size, args.salience_check_samples)]
    batch = [train_dataset[i] for i in batch_idx]
    batch = [x[0] for x in batch]
    batch_loader = DataLoader(batch, args.batch_size)

    if not student_model.shrink_layer(bottleneck_layer_idx, batch_loader, factor=args.shrink_factor):
        return student_model
    torch.cuda.ipc_collect()
    print('\nchange in mem after shrinking:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))
    student_model = student_model.to(args.device)
    
    bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
    print(student_model, bottleneck_size)   
    delta = 0
    break_next_iter = False
    while (bottleneck_size > 0):
        val_metric = evaluate(student_model, val_dataset, args, val_metric_fn)
        print('val_metric:', val_metric)

        print('bottleneck_size = %d' % bottleneck_size)
        logger.info('bottleneck_size = %d' % bottleneck_size)        
        gpu_mem = torch.cuda.memory_allocated(args.device)
        metric = distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn)
        torch.cuda.ipc_collect()
        
        print('\nchange in mem after distillation:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))
        logger.info('current metric = %.4f base_metric = %.4f' % (metric, base_metric))

        delta = metric - base_metric
        if args.fine_tune and delta < args.tol:
            metric = fine_tune()
        delta = metric - base_metric
        if delta < args.tol:            
            logger.info('tolerance violated; stopping distillation')            
            break
        else:
            logger.info('saving model...')
            torch.save(student_model, args.outfile)
            if args.train_on_student:
                train_logits = get_logits(student_model, train_dataset, args)
                train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=True)

        gpu_mem = torch.cuda.memory_allocated(args.device)
        if break_next_iter or not student_model.shrink_layer(bottleneck_layer_idx, batch_loader, factor=args.shrink_factor):
            if break_next_iter:
                break
            else:
                break_next_iter = True
        student_model = student_model.to(args.device)
        torch.cuda.ipc_collect()
        print('\nchange in mem after shrinking:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))

        new_bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
        if new_bottleneck_size == bottleneck_size:
            break
        else:
            bottleneck_size = new_bottleneck_size
    
    student_model = student_model.to(args.device) 
    
    val_metric = evaluate(student_model, val_dataset, args, val_metric_fn)
    print('val_metric:', val_metric)
    logger.info('val_metric:%f'%val_metric)
    delta = val_metric - base_metric
    if delta >= args.tol:
        # logger.info('saving model...')        
        torch.save(student_model, args.outfile)

    student_model = torch.load(args.outfile).to(args.device)    
    return student_model

def simple_distillation(teacher_model: models.ModelWrapper, student_model, train_dataset, val_dataset, nclasses, args):
    base_metric = evaluate(teacher_model, val_dataset, args)
    print('base_metric = %.4f' % (base_metric))

    train_logits = get_logits(teacher_model, train_dataset, args)
    train_loader = DataLoader(train_logits, args.batch_size, num_workers=cpu_count()//2, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=False)

    distill(student_model,train_loader,val_loader,base_metric, args) 

def main(args):
    train_dataset, test_dataset, nclasses = utils.get_datasets(args)
    new_test_size = int(0.8*len(test_dataset))
    val_size = len(test_dataset) - new_test_size
    test_dataset, val_dataset = random_split(test_dataset, [new_test_size, val_size])
    
    teacher_model = torch.load(args.teacher_model_file).to(args.device) 
    teacher_model = teacher_model.eval()
    
    base_metric = evaluate(teacher_model, val_dataset, args)
    print('base_metric = %.4f' % (base_metric))
    logger.info('base_metric = %.4f' % (base_metric))
    if args.test_only:
        return
    train_logits = get_logits(teacher_model, train_dataset, args)
    teacher_model = teacher_model.cpu()
    
    if args.student_model_file is None:
        student_model = copy_model(teacher_model, args.device, reinitialize=(not args.retain_teacher_weights))
        if args.predictive_pruning:
            student_model = StudentModelWrapper2(student_model, logger, args)
        else:
            student_model = StudentModelWrapper(student_model, logger)
        for param in student_model.parameters():
            param.requires_grad = True
    else:
        student_model = torch.load(args.student_model_file)
    del teacher_model

    student_model = student_model.to(args.device)
    base_metric = evaluate(student_model, val_dataset, args)
    print('student_base_metric = %.4f' % (base_metric))
    logger.info('student_base_metric = %.4f' % (base_metric))
    
    shrinkable_layers = student_model.get_shrinkable_layers()
    if args.reverse_shrink_order:
        shrinkable_layers = shrinkable_layers[::-1]
    if args.shrink_all or len(args.shrink_layer_idxs) > 0:  
        if len(args.shrink_layer_idxs) > 0:
            not_shrinkables = [i for i in shrinkable_layers if i not in args.shrink_layer_idxs]
            shrinkable_layers = args.shrink_layer_idxs
        else:
            not_shrinkables = args.exclude_layers            
        while len(shrinkable_layers) > 0:
            student_model = iterative_distillation(student_model, shrinkable_layers[0], train_logits, val_dataset, test_dataset, nclasses, base_metric, args, mLogger=logger)
            if args.train_on_student:
                train_logits = get_logits(student_model, train_dataset, args)
            print(student_model)
            new_shrinkable_layers = student_model.get_shrinkable_layers(not_shrinkables)
            if args.reverse_shrink_order:
                new_shrinkable_layers = new_shrinkable_layers[::-1]
            if shrinkable_layers == new_shrinkable_layers:
                not_shrinkables.append(shrinkable_layers[0])            
            shrinkable_layers = [x for x in new_shrinkable_layers if x not in not_shrinkables]            
            print(not_shrinkables, shrinkable_layers)
    else:
        student_model = iterative_distillation(student_model, shrinkable_layers[-1], train_logits, val_dataset, test_dataset, nclasses, base_metric, args, mLogger=logger)
    

    print(student_model)
    test_metric = evaluate(student_model, test_dataset, args)
    print('test_metric = %.4f' % (test_metric))
    logger.info('test_metric = %.4f' % (test_metric))
    torch.save(student_model, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--teacher_model_file', type=str)
    parser.add_argument('--student_model_file', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--n_fine_tune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--salience_check_samples', type=int, default=256)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam/sgd')
    parser.add_argument('--soft_loss_type', type=str, default='xent', help='xent/rmse')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--C', type=float, default=0)
    parser.add_argument('--T', type=float, default=1)
    parser.add_argument('--loss_weight', type=float, default=1e-4)
    parser.add_argument('--max_T', type=float, default=100)
    parser.add_argument('--tol', type=float, default=-0.05)
    parser.add_argument('--shrink_factor', type=float, default=0.9)
    parser.add_argument('--outfile', type=str, default='models/distilled/distilled_model.pt')
    parser.add_argument('--logfile', type=str, default='logs/distillation/distillation.log')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--retain_teacher_weights', action='store_true')
    parser.add_argument('--shrink_all', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--reverse_shrink_order', action='store_true')
    parser.add_argument('--train_on_student', action='store_true')
    parser.add_argument('--shrink_layer_idxs', nargs='+', type=int, default=[])
    parser.add_argument('--exclude_layers', nargs='+', type=int, default=[])
    parser.add_argument('--increase_loss_weight', action='store_true')
    parser.add_argument('--anneal_temperature', action='store_true')
    parser.add_argument('--predictive_pruning', action='store_true')
    parser.add_argument('--readjust_weights', action='store_true')    
    parser.add_argument('--test_only', action='store_true')
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

    logger.info(args)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)