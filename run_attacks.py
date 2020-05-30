import os
import argparse
import sys

def main(args):
    if args.attack == 'pgdl2' or args.attack == 'cwl2':
        norm = 'L2'
        eps = [1.0, 2.0] 
    elif args.attack == 'pgdinf':
        norm = 'Linf'
        eps = [x/255 for x in [8, 16]]        
    elif args.attack == 'cwinf':
        eps = [x/255 for x in [8, 16]]
    elif args.attack == 'jsma':
        eps = [x/255 for x in [127, 255]]
    else:
        norm = 'none'
        eps = eps_step = [0]
    eps_step = [x/4 for x in eps]
    
    for i in range(len(eps)):
        # if args.nb_restarts > 1:
        # cmd = "CUDA_VISIBLE_DEVICES='%s'  python evaluate_on_pgd_attacks.py --model_path %s --eps %f --eps_iter %f --norm %s --nb_restart %d" % (os.environ['CUDA_VISIBLE_DEVICES'], args.model_path, eps[i], eps_step[i], norm, args.nb_restarts)
        # else:
        cmd = "CUDA_VISIBLE_DEVICES='%s'  python attack_classifier.py %s --dataset %s --attack %s --eps %f --eps_iter %f --nb_iter 100 --nb_restart %d --batch_size %d" % (os.environ['CUDA_VISIBLE_DEVICES'], args.model_path, args.dataset, args.attack, eps[i], eps_step[i], args.nb_restarts, args.batch_size)
        # cmd = "CUDA_VISIBLE_DEVICES='%s'  python attack_classifier_art.py %s --dataset %s --attack %s --eps %f --eps_iter %f --nb_iter 100 --nb_restart %d" % (os.environ['CUDA_VISIBLE_DEVICES'], args.model_path, args.dataset, args.attack, eps[i], eps_step[i], args.nb_restarts)
        if args.binary:
            cmd += ' --binary_classification'
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('attack')
    parser.add_argument('dataset')
    parser.add_argument('--nb_restarts', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--binary', action='store_true')
    args = parser.parse_args()

    main(args)