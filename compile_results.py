import os
import torch
from torch import device
from torch import nn
from argparse import Namespace
from distillation import main as test
from distillation import StudentModelWrapper, StudentModelWrapper2
import json
from datetime import datetime

def get_expt_info(logfn):
    info = {k:None for k in ['datetime', 'ns', 'test_metric', 'val_metric', 'base_metric']}
    with open(logfn) as f:
        txt = f.readlines()
        if 'test_metric' not in txt[-1]:
            return None

        for i in range(len(txt)-1,-1, -1):
            line = txt[i]
            if "Namespace" in line:
                split = txt[i].split()
                date = split[0]
                time = split[1]
                ns_idx = txt[i].index('Namespace')                
                ns = eval(txt[i][ns_idx:])

                info['datetime'] = "%s %s" % (date,time)
                info['ns'] = ns
                break

            elif 'test_metric' in line and  info['test_metric'] is None:
                info['test_metric'] = float(line.split('=')[-1])
            elif 'val_metric' in line and info['val_metric'] is None:
                info['val_metric'] = float(line.split(':')[-1])
            elif 'base_metric' in line and  info['base_metric'] is None:
                info['base_metric'] = float(line.split('=')[-1])

        return info

def get_num_params(model_path):
    model = torch.load(model_path)
    num_params = sum([p.numel() for p in model.parameters()])
    num_dense = sum([sum([p.numel() for p in m.parameters()]) for m in model.modules() if isinstance(m,nn.Linear)])
    num_conv = sum([sum([p.numel() for p in m.parameters()]) for m in model.modules() if isinstance(m,nn.Conv2d)])
    return {
        'num_params':num_params, 
        'num_dense':num_dense, 
        'num_conv':num_conv
    }

model_dir = 'models/distilled/'
log_dir = 'logs/distilled/'

log_files = os.listdir(log_dir)
log_files = [os.path.join(log_dir, x) for x in log_files]

results = []

for lf in log_files:
    expt_info = get_expt_info(lf)
    
    if expt_info is None:
        continue

    ns = expt_info['ns']

    if hasattr(ns, 'retain_teacher_weights') and ns.retain_teacher_weights == True:
        print(ns.outfile)                
        ns.test_only = True
        ns.log_file = '/dev/null'
        ns.student_model_file = ns.outfile
        ns.outfile = '/dev/null'
        ns.datafolder = '/home/mshah1/workhorse3'
        
        arg_dict = dict(vars(ns))

        if ns.cuda and torch.cuda.is_available():
            ns.device = torch.device('cuda')
        else:
            ns.device = torch.device('cpu')        

        if os.path.exists(ns.teacher_model_file) and os.path.exists(ns.student_model_file):
            # base_metric, metric = test(ns)
            teacher_params = get_num_params(ns.teacher_model_file)
            student_params = get_num_params(ns.student_model_file)
        else:
            continue
        reduction = {k: 1 - student_params[k]/teacher_params[k] for k in teacher_params}

        expt_info.update({
            'model_name': lf.split('/')[-1].split('.')[0],
            'datetime': expt_info['datetime'],
            'teacher_num_params':teacher_params,
            'student_num_params':student_params,
            'reduction': reduction,
            'args': arg_dict
        })
        expt_info.pop('ns')
        results.append(expt_info)
        # if len(results) == 3:
        #     break
    torch.cuda.ipc_collect()
results = sorted(results, key=lambda x: x['model_name'], reverse=True)

now = datetime.now()
dt = now.strftime("%m%d%Y")
with open('compiled_results_%s.json'%dt, 'w') as f:
    json.dump(results, f)