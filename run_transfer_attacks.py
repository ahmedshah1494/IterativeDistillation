from subprocess import Popen
from attack_classifier import transfer_attack
import os
import time
import numpy as np
import itertools
source_data = [
    'models/baseline/alexnet_cifar10.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_roundRobin_predictive_pruning_adjustingW_postActivation_distilled.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/baseline/vgg16_cifar10.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',    
    'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled_tol03.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/distilled/vgg16_cifar10_allLayers_reverse_roundRobin_predictive_pruning_scoreByResidualMI_noReg_adjustingW_postActivation_distilled_tol-03.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
    'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled_tol-00.ptadvdata_pgdinf_eps=0.031373_1restarts.pt',
]
target_models = [
    'models/baseline/alexnet_cifar10.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_roundRobin_predictive_pruning_adjustingW_postActivation_distilled.pt',
    'models/distilled/alexnet_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled.pt',
    'models/baseline/vgg16_cifar10.pt',    
    'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_scaleByLossGrad_noAbs_adjustingW_postActivation_distilled_tol03.pt',
    'models/distilled/vgg16_cifar10_allLayers_reverse_roundRobin_predictive_pruning_scoreByResidualMI_noReg_adjustingW_postActivation_distilled_tol-03.pt',
    'models/distilled/vgg16_cifar10_allLayers_reverse_predictive_pruning_adjustingW_postActivation_distilled_tol-00.pt',
]

for p in itertools.chain(source_data, target_models):
    if not os.path.exists(p):
        raise FileNotFoundError(p)

gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
curr_gpu_idx = 0
proc_list = []
for sd in source_data:
    for model in target_models:
        cmd = 'python attack_classifier.py %s --dataset %s --transfer_attack' % (model, sd)
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