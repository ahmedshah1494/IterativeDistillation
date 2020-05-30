import sys
import os

model_path=sys.argv[1]
dataset=sys.argv[2]
for attack in ['pgdinf', 'pgdl2']:
    cmd="CUDA_VISIBLE_DEVICES='%s' python run_attacks.py %s %s %s --nb_restarts 4" % (os.environ['CUDA_VISIBLE_DEVICES'], model_path, attack, dataset)
    os.system(cmd)
    