import os
import argparse
from subprocess import Popen
import time
import sys
import torch
from torch import nn
import numpy as np
from distillation import StudentModelWrapper, StudentModelWrapper2
from attack_classifier import EnsembleWrapper        

def get_cmdline(args):
    if args.attack == 'pgdl2' or args.attack == 'cwl2':
        norm = 'L2'
        if args.dataset == 'cifar10':
            eps = [0.5, 1.0, 2.0] 
        if args.dataset == 'mnist':
            eps = [2.0, 3.0, 4.0] 
        eps_step = [x/4 for x in eps]
    elif args.attack == 'pgdinf':
        norm = 'Linf'
        if args.dataset == 'cifar10':
            eps = [x/255 for x in [4, 8, 16]]
            eps_step = [x/4 for x in eps]
        if args.dataset == 'mnist':
            eps = [0.2, 0.3, 0.4]
            eps_step = [x/40 for x in eps]
    elif args.attack == 'cwinf':
        if args.dataset == 'cifar10':
            eps = [x/255 for x in [8, 16]]
        else:
            raise NotImplementedError
        eps_step = [x/4 for x in eps]
    elif args.attack == 'jsma':
        if args.dataset == 'cifar10':
            eps = [x/255 for x in [127, 255]]
        else:
            raise NotImplementedError
        eps_step = [x/4 for x in eps]
    else:
        norm = 'none'
        eps = eps_step = [0]    
    cmdlines = []
    for i in range(len(eps)):
        cmds = [
            "python evaluate_on_pgd_attacks.py --model_path %s --eps %f --eps_iter %f --norm %s --nb_restart %d" % (args.model_path, eps[i], eps_step[i], norm, args.nb_restarts),
            "python attack_classifier.py %s --dataset %s --attack %s --eps %f --eps_iter %f --nb_iter 100 --nb_restart %d --batch_size %d" % (args.model_path, args.dataset, args.attack, eps[i], eps_step[i], args.nb_restarts, args.batch_size),
            "python attack_classifier_art.py %s --dataset %s --attack %s --eps %f --eps_iter %f --nb_iter 100 --nb_restart %d" % (args.model_path, args.dataset, args.attack, eps[i], eps_step[i], args.nb_restarts)
        ]        
        cmd = cmds[args.attack_library]
        if args.binary:
            cmd += ' --binary_classification'   
        cmdlines.append(cmd)
    return cmdlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',)
    parser.add_argument('--attack', nargs='+')
    parser.add_argument('--dataset', choices=('cifar10', 'mnist'))
    parser.add_argument('--nb_models', type=int)
    parser.add_argument('--nb_restarts', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--attack_library', default=1, type=int)
    parser.add_argument('--consensus_pc', default=1., type=float)
    parser.add_argument('--binary', action='store_true')
    args = parser.parse_args()

    # ensemble_tmp_fp = 'models/.ensemble.%f.pt' % time.time()

    # model_paths = args.model_path
    vgg_ensemble = [        
        'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled_tol03.pt',
        'models/distilled/vgg16_cifar10_allLayers_reverse_roundRobin_predictive_pruning_scoreByResidualMI_noReg_adjustingW_postActivation_distilled_tol-03.pt',
        'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled_tol-00.pt',
    ]
    vgg_alexnet_ensemble = [
        'models/baseline/alexnet_cifar10.pt',
        'models/baseline/vgg16_cifar10.pt',    
        'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled.pt',
        'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled_tol03.pt',
        'models/distilled/alexnet_cifar10_allLayers_reverse_roundRobin_predictive_pruning_adjustingW_postActivation_distilled.pt',
        'models/distilled/vgg16_cifar10_allLayers_reverse_roundRobin_predictive_pruning_scoreByResidualMI_noReg_adjustingW_postActivation_distilled_tol-03.pt',
        'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled.pt',                                
        'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled_tol-00.pt',
    ]    
    if args.model == 'vgg_ensemble':
        model_paths = vgg_ensemble
        if args.nb_models is None:
            args.nb_models = len(model_paths)
        outfile = 'models/ensembles/vgg_ensemble_%dm_%fcp.pt' % (args.nb_models, args.consensus_pc)
        print('evaluating vgg ensemble with %d models...' % args.nb_models)
    if args.model == 'vgg-alexnet_ensemble':
        model_paths = vgg_alexnet_ensemble
        if args.nb_models is None:
            args.nb_models = len(model_paths)
        outfile = 'models/ensembles/vgg-alexnet_ensemble_%dm_%fcp.pt' % (args.nb_models, args.consensus_pc)
        print('evaluating vgg-alexnet ensemble with %d models...' % args.nb_models)
    
    model_paths = model_paths[:args.nb_models]
    models = [torch.load(mp) for mp in model_paths]
    ensemble = EnsembleWrapper(models, args.consensus_pc)

    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    torch.save(ensemble, outfile)

    attacks = args.attack

    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    curr_gpu_idx = 0
    proc_list = []
        
    for attack in attacks:
        args.model_path = outfile
        args.attack = attack
        cmds = get_cmdline(args)
        for cmd in cmds:             
            while len(proc_list) >= len(gpus):
                for i,p in enumerate(proc_list):
                    if p.poll() is not None:
                        proc_list.pop(i)
                
                if len(proc_list) >= len(gpus):
                    time.sleep(10)
                else:
                    break
            
            print('gpu_id:',gpus[curr_gpu_idx], cmd)
            p = Popen(cmd.split(), env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[curr_gpu_idx]))
            proc_list.append(p)        
            curr_gpu_idx = (curr_gpu_idx + 1) % len(gpus)              

    for p in proc_list:
        p.wait()