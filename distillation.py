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
import pickle
import itertools

class StudentModelWrapper(nn.Module, models.ModelWrapper):
    def __init__(self, model, logger, args):
        super(StudentModelWrapper, self).__init__()        
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

def distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn=utils.compute_correct, logger=None):    
    criterion = distillation_loss
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise ValueError('optimizer should be either "adam" or "sgd"')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, factor=0.5)

    best_model = None
    t = range(args.nepochs)
    T = args.T
    loss_w = args.loss_weight

    best_acc = 0
    epochs_since_best = 0
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

        if val_acc - best_acc > 1e-5:
            best_acc = val_acc
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best > 3*args.patience and args.early_stop:
                break

        if optimizer.param_groups[0]['lr'] <= 1e-7 and args.early_stop:
            break
        
        if args.increase_loss_weight:
            loss_w += (1-args.loss_weight) / args.nepochs
        if args.anneal_temperature:
            T -= (args.T-1) / args.nepochs

        # t.set_postfix(train_loss=epoch_loss, train_soft_loss=epoch_soft_loss, train_hard_loss=epoch_hard_loss, 
        #                 val_acc=val_acc, lr=optimizer.param_groups[0]['lr'], T=T, lossW=loss_w)        
        # if i % 10 == 0:
        print('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4E' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
        if logger is not None:
            logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4E' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
        if val_acc-base_metric >= args.tol:
            if logger is not None:
                logger.info('epoch#%d train_loss=%.3f train_hard_loss=%.3f train_soft_loss=%.3f val_acc=%.3f lr=%.4E' % (i, epoch_loss, epoch_hard_loss, epoch_soft_loss, val_acc, optimizer.param_groups[0]['lr']))
            break
    return val_acc
    
def iterative_distillation(student_model:StudentModelWrapper, bottleneck_layer_idx, train_dataset, 
                            val_dataset, test_dataset, nclasses, base_metric, args, mLogger=None, 
                            val_metric_fn=utils.compute_correct): 
    if mLogger is not None:
        logger = mLogger
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//4, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//4, shuffle=False)

    def fine_tune():
        loss_w = args.loss_weight            
        args.loss_weight = 1

        nepochs = args.nepochs
        args.nepochs = args.n_fine_tune_epochs        

        metric = distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn, mLogger)

        args.loss_weight = loss_w
        args.nepochs = nepochs
        
        torch.cuda.ipc_collect()
        return metric

    print(student_model)

    if args.tune_current_layer_only:
        for li, l in enumerate(student_model.layers):
            if li != bottleneck_layer_idx:
                for p in l.parameters():
                    p.requires_grad = False 

    if args.store_Ys:
        Ys = []
        y = student_model.get_Y(val_loader, bottleneck_layer_idx)
        Ys.append(y)
    
    torch.save(student_model, args.outfile)
    student_model = student_model.to(args.device)

    print('shrinking layer %d...' % bottleneck_layer_idx)    
    logger.info('shrinking layer %d...' % bottleneck_layer_idx)

    gpu_mem = torch.cuda.memory_allocated(args.device)
    
    bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
    # if student_model.is_last_conv(bottleneck_layer_idx):
    #     i = bottleneck_layer_idx+1
    #     while i < len(student_model.layers) and not utils.is_weighted(student_model.layers[i]) and not isinstance(student_model.layers[i], nn.Dropout):
    #         i += 1
    #     _, output_size = utils.get_layer_input_output_size(student_model.layers[i])
    #     print(output_size, output_size//bottleneck_size)
    #     batch_idx = np.random.permutation(np.arange(len(train_dataset)))[:max(output_size, args.salience_check_samples)]
    # else:
    batch_idx = np.random.permutation(np.arange(len(train_dataset)))[:max(bottleneck_size, args.salience_check_samples)]
    
    batch = [train_dataset[i] for i in batch_idx]
    # batch = [x[0] for x in batch]
    batch_loader = DataLoader(batch, args.batch_size)
    
    torch.cuda.ipc_collect()

    if not student_model.shrink_layer(bottleneck_layer_idx, batch_loader, factor=args.shrink_factor):
        print('returning...')
        return student_model
        
    torch.cuda.ipc_collect()
    print('\nchange in mem after shrinking:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))
    student_model = student_model.to(args.device)
    
    bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
    print(student_model, bottleneck_size)   
    delta = 0
    break_next_iter = False
    tuned = False 
    while (bottleneck_size > 0):
        metric = evaluate(student_model, val_dataset, args, val_metric_fn)
        print('val_metric:', metric)
        logger.info('val_metric: %0.4f' % metric)

        delta = metric - base_metric
        
        print('bottleneck_size = %d' % bottleneck_size)
        logger.info('bottleneck_size = %d' % bottleneck_size)
        
        if delta < args.tol:
            tuned = True            
            gpu_mem = torch.cuda.memory_allocated(args.device)
            metric = distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn, mLogger)
            torch.cuda.ipc_collect()
            
            print('\nchange in mem after distillation:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))
            logger.info('current metric = %.4f base_metric = %.4f' % (metric, base_metric))
        delta = metric - base_metric

        if args.store_Ys:
            if len(Ys) < args.store_Y_iters:
                y = student_model.get_Y(val_loader, bottleneck_layer_idx)
                Ys.append((y, tuned))
                tuned = False                
                with open(args.Y_outfile, 'wb') as f:
                    pickle.dump(Ys, f)
            if len(Ys) == args.store_Y_iters:                
                return student_model

        if args.fine_tune and delta < args.tol:
            metric = fine_tune()
        delta = metric - base_metric
        if delta < args.tol:            
            logger.info('tolerance violated; stopping distillation')            
            break
        else:
            print(student_model, bottleneck_size)   
            print('saving model...')
            logger.info('saving model...')
            torch.save(student_model, args.outfile)
            if args.train_on_student:
                train_logits = get_logits(student_model, train_dataset, args)
                train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=True)

        gpu_mem = torch.cuda.memory_allocated(args.device)

        if args.round_robin:
            break

        if break_next_iter or not student_model.shrink_layer(bottleneck_layer_idx, batch_loader, factor=args.shrink_factor):
            if break_next_iter:
                break
            else:
                break_next_iter = True                

        student_model = student_model.to(args.device)
        torch.cuda.ipc_collect()
        print('\nchange in mem after shrinking:%d\t%d\n' % (torch.cuda.memory_allocated(args.device) - gpu_mem, torch.cuda.memory_allocated()))

        new_bottleneck_size,_ = utils.get_layer_input_output_size(student_model.layers[bottleneck_layer_idx])
        print('new_bottleneck_size:', new_bottleneck_size)
        if new_bottleneck_size == bottleneck_size:
            break
        else:
            bottleneck_size = new_bottleneck_size                
    
    student_model = student_model.to(args.device) 
    
    val_metric = evaluate(student_model, val_dataset, args, val_metric_fn)
    print('val_metric:', val_metric)
    logger.info('val_metric:%f'%val_metric)
    student_model = student_model.cpu()

    delta = val_metric - base_metric
    if delta >= args.tol:
        print(student_model, bottleneck_size)
        logger.info('saving model...')        
        torch.save(student_model, args.outfile)
    del student_model
    
    student_model = torch.load(args.outfile).to(args.device)
    val_metric = evaluate(student_model, val_dataset, args, val_metric_fn)
    print('val_metric:', val_metric)
    logger.info('val_metric:%f'%val_metric)
    return student_model

def global_compression(student_model:StudentModelWrapper, train_dataset, 
                            val_dataset, test_dataset, nclasses, base_metric, 
                            shrinkable_layers, args, mLogger=None, 
                            val_metric_fn=utils.compute_correct):
    if mLogger is not None:
        logger = mLogger
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=cpu_count()//4, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//4, shuffle=False)

    torch.save(student_model, args.outfile)
    student_model = student_model.to(args.device)

    metric = evaluate(student_model, val_dataset, args, val_metric_fn)
    print('val_metric:', metric)
    logger.info('val_metric: %0.4f' % metric)

    delta = metric - base_metric
    print(student_model)
    while delta >= args.tol:           
        print('saving model...')
        logger.info('saving model...')
        torch.save(student_model, args.outfile)

        neuron_scores = []
        As = {}
        mean_Zs = {}
        for layer_idx in shrinkable_layers:
            layer_size,_ = utils.get_layer_input_output_size(student_model.layers[layer_idx])
            batch_idx = np.random.permutation(np.arange(len(train_dataset)))[:max(layer_size, args.salience_check_samples)]
        
            batch = [train_dataset[i] for i in batch_idx]
            # batch = [x[0] for x in batch]
            batch_loader = DataLoader(batch, args.batch_size)
            A, layer_neuron_scores, mean_z = student_model.compute_prune_probability(layer_idx, batch_loader, normalize=False)
            A = A.detach().cpu()
            neuron_scores.append(layer_neuron_scores)
            As[layer_idx] = A
            mean_Zs[layer_idx] = mean_z
        neuron_layer_idxs = [[i]*len(l) for i,l in zip(shrinkable_layers, neuron_scores)]
        neuron_layer_idxs = list(itertools.chain(*neuron_layer_idxs))

        neuron_idxs = [range(len(l)) for l in neuron_scores]
        neuron_idxs = list(itertools.chain(*neuron_idxs))

        neuron_scores = list(itertools.chain(*neuron_scores))
        print(len(neuron_layer_idxs), len(neuron_idxs), len(neuron_scores))
        sorted_neurons = sorted(zip(neuron_layer_idxs, neuron_idxs, neuron_scores), 
                                key=lambda x: x[2],
                                reverse=True)
        pruned_neurons = sorted_neurons[:int((1-args.shrink_factor)*len(sorted_neurons))]
        layer2neuron = {}
        for li,ni,_ in pruned_neurons:
            layer2neuron.setdefault(li,[]).append(ni)
        print(layer2neuron)
        for li in sorted(layer2neuron.keys(), reverse=True):
            student_model.shrink_layer(li, batch_loader, 
                                        pruned_neurons=layer2neuron[li], 
                                        A=As.pop(li).to(student_model.device), 
                                        mean_Z=mean_Zs[li]
                                        )
        print(student_model)
        metric = evaluate(student_model, val_dataset, args, val_metric_fn)
        print('val_metric:', metric)
        logger.info('val_metric: %0.4f' % metric)

        delta = metric - base_metric
        if delta < args.tol:
            metric = distill(student_model, train_loader, val_loader, base_metric, args, val_metric_fn, mLogger)
            torch.cuda.ipc_collect()
            print('current metric = %.4f base_metric = %.4f' % (metric, base_metric))
            logger.info('current metric = %.4f base_metric = %.4f' % (metric, base_metric))
        delta = metric - base_metric
    
    student_model = torch.load(args.outfile).to(args.device)
    val_metric = evaluate(student_model, val_dataset, args, val_metric_fn)
    print('val_metric:', val_metric)
    logger.info('val_metric:%f'%val_metric)
    return student_model

def simple_distillation(teacher_model: models.ModelWrapper, student_model, train_dataset, val_dataset, nclasses, args):
    base_metric = evaluate(teacher_model, val_dataset, args)
    print('base_metric = %.4f' % (base_metric))

    train_logits = get_logits(teacher_model, train_dataset, args)
    train_loader = DataLoader(train_logits, args.batch_size, num_workers=cpu_count()//2, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=cpu_count()//2, shuffle=False)

    distill(student_model,train_loader,val_loader,base_metric, args) 

def fine_tune(student_model, train_loader, val_loader):
    loss_w = args.loss_weight            
    args.loss_weight = 1

    nepochs = args.nepochs
    args.nepochs = args.n_fine_tune_epochs        

    metric = distill(student_model, train_loader, val_loader, 1.0, args, utils.compute_correct, logger)

    args.loss_weight = loss_w
    args.nepochs = nepochs
    
    torch.cuda.ipc_collect()
    return metric

def main(args):
    train_dataset, test_dataset, nclasses = utils.get_datasets(args, not args.no_normalization)
    new_test_size = int(0.8*len(test_dataset))
    val_size = len(test_dataset) - new_test_size
    test_dataset, val_dataset = random_split(test_dataset, [new_test_size, val_size])
    
    teacher_model = torch.load(args.teacher_model_file).to(args.device) 
    teacher_model = teacher_model.eval()
    
    teacher_base_metric = evaluate(teacher_model, val_dataset, args)
    teacher_test_metric = evaluate(teacher_model, test_dataset, args)
    if not args.test_only:
        train_logits = get_logits(teacher_model, train_dataset, args)
        teacher_model = teacher_model.cpu()
    
    if args.student_model_file is None:
        student_model = copy_model(teacher_model, args.device, reinitialize=(not args.retain_teacher_weights))
        if args.predictive_pruning:
            student_model = StudentModelWrapper2(student_model, logger, args)
        else:
            student_model = StudentModelWrapper(student_model, logger, args)
        for param in student_model.parameters():
            param.requires_grad = True
    else:
        student_model = torch.load(args.student_model_file)
    del teacher_model

    student_model.args = args
    student_model = student_model.to(args.device)
    student_base_metric = evaluate(student_model, val_dataset, args)

    if args.test_only:
        student_test_metric = evaluate(student_model, test_dataset, args)
        print('teacher_test_metric = %.4f' % (teacher_test_metric))
        print('teacher_base_metric = %.4f' % (teacher_base_metric))
        # logger.info('base_metric = %.4f' % (teacher_base_metric))
        print('student_base_metric = %.4f' % (student_base_metric))
        # logger.info('student_base_metric = %.4f' % student_base_metric)
        return teacher_base_metric, student_base_metric, teacher_test_metric, student_test_metric

    base_metric = max(args.base_metric, teacher_base_metric)
    
    print('teacher_base_metric = %.4f' % (teacher_base_metric))
    logger.info('base_metric = %.4f' % (teacher_base_metric))
    print('student_base_metric = %.4f' % (student_base_metric))
    logger.info('student_base_metric = %.4f' % student_base_metric)
    print('base_metric = %.4f' % (base_metric))
    logger.info('base_metric = %.4f' % base_metric)
    
    shrinkable_layers = student_model.get_shrinkable_layers() 
    if args.global_pruning:
        student_model = global_compression(student_model, train_logits, val_dataset, test_dataset, nclasses, base_metric, shrinkable_layers, args, mLogger=logger)
    else:
        not_shrinkables = []       
        if args.reverse_shrink_order:
            shrinkable_layers = shrinkable_layers[::-1]
        if args.shrink_all or len(args.shrink_layer_idxs) > 0:  
            old_num_params = num_params = sum([p.numel() for p in student_model.parameters()])        
            # rr_iters = args.round_robin_iters if args.round_robin else 1
            # for rri in range(rr_iters):
            rri = 0

            shrinkable_layers_ = shrinkable_layers        

            while rri == 0 or old_num_params != num_params:
                student_model.reset()

                shrinkable_layers = shrinkable_layers_                    

                if len(args.shrink_layer_idxs) > 0:
                    not_shrinkables = [i for i in shrinkable_layers if i not in args.shrink_layer_idxs]
                    shrinkable_layers = args.shrink_layer_idxs
                else:
                    not_shrinkables = args.exclude_layers[:]
                
                if rri == 0 and args.start_layer_idx >= 0 and args.start_layer_idx in shrinkable_layers:
                    if args.reverse_shrink_order:
                        not_shrinkables += [i for i in shrinkable_layers if i > args.start_layer_idx]
                    else:
                        not_shrinkables += [i for i in shrinkable_layers if i < args.start_layer_idx]
                    shrinkable_layers = shrinkable_layers[shrinkable_layers.index(args.start_layer_idx):]                        
                        
                print(rri, not_shrinkables, shrinkable_layers, args.exclude_layers, args.shrink_layer_idxs)            
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
                
                if not args.round_robin:
                    break

                old_num_params = num_params
                num_params = sum([p.numel() for p in student_model.parameters()])
                num_dense = sum([sum([p.numel() for p in m.parameters()]) for m in student_model.modules() if isinstance(m,nn.Linear)])
                num_conv = sum([sum([p.numel() for p in m.parameters()]) for m in student_model.modules() if isinstance(m,nn.Conv2d)])
                print('change in num_params: %d -> %d' % (old_num_params, num_params))
                logger.info('num params: %d' % num_params)
                logger.info('num dense: %d' % num_dense)
                logger.info('num conv: %d' % num_conv)
                rri += 1
                
        else:
            student_model = iterative_distillation(student_model, shrinkable_layers[args.start_layer_idx], train_logits, val_dataset, test_dataset, nclasses, base_metric, args, mLogger=logger)

    print(student_model)
    test_metric = evaluate(student_model, test_dataset, args)
    print('test_metric = %.4f' % (test_metric))
    print('teacher_test_metric = %.4f' % (teacher_test_metric))
    logger.info('test_metric = %.4f' % (test_metric))
    logger.info('teacher_test_metric = %.4f' % (teacher_test_metric))
    torch.save(student_model, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--datafolder', type=str,default="/home/mshah1/workhorse3")
    parser.add_argument('--teacher_model_file', type=str)
    parser.add_argument('--student_model_file', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--n_fine_tune_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--salience_check_samples', type=int, default=256)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--base_metric', type=float, default=0.0)
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
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--retain_teacher_weights', action='store_true')
    parser.add_argument('--shrink_all', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--fine_tune_only', action='store_true')
    parser.add_argument('--reverse_shrink_order', action='store_true')
    parser.add_argument('--round_robin', action='store_true')
    parser.add_argument('--round_robin_iters', type=int, default=10)
    parser.add_argument('--train_on_student', action='store_true')
    parser.add_argument('--shrink_layer_idxs', nargs='+', type=int, default=[])
    parser.add_argument('--start_layer_idx', type=int, default=-1)
    parser.add_argument('--exclude_layers', nargs='+', type=int, default=[])
    parser.add_argument('--increase_loss_weight', action='store_true')
    parser.add_argument('--anneal_temperature', action='store_true')
    parser.add_argument('--predictive_pruning', action='store_true')
    parser.add_argument('--predictive_pruning_iters', type=int, default=1000)
    parser.add_argument('--predictive_pruning_batch_size', type=int, default=128)
    parser.add_argument('--predictive_pruning_lr', type=float, default=1e-3)
    parser.add_argument('--tune_current_layer_only', action='store_true')
    parser.add_argument('--store_Ys', action='store_true')
    parser.add_argument('--store_Y_iters', type=int, default=3)
    parser.add_argument('--Y_outfile', type=str, default='')
    parser.add_argument('--readjust_weights', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--no_normalization', action='store_true')
    parser.add_argument('--scale_by_grad', type=str, default='none', choices=('none', 'output', 'loss'))
    parser.add_argument('--global_pruning', action='store_true')
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
    print(args.device)
    main(args)