import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from traitlets.config.loader import PyFileConfigLoader
import torchvision
from torch.utils.data import DataLoader, Subset
from models import ModelWrapper, ModelWrapper2
from distillation import StudentModelWrapper, StudentModelWrapper2
import argparse
import re
import utils
from advertorch_examples.utils import get_test_loader
import os

import sys
from pytorch_lightning import Trainer

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, FGSM, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ChooseBestAttack
from advertorch.attacks.utils import attack_whole_dataset
from advertorch_examples.utils import get_cifar10_test_loader, get_test_loader
from advertorch.utils import get_accuracy
from advertorch.loss import CWLoss

class NullAdversary:
    def __init__(self,*args,**kwargs):
        pass 
    def perturb(self,x,y):
        return x

class NormalizationWrapper(nn.Module):
    def __init__(self, model):
        super(NormalizationWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        x = utils.normalize_image_tensor(x)
        return self.model(x)

class Attacker:
    def __init__(self,source_model, dataloader, attack_class, *args, binary_classification=False, max_instances=-1, **kwargs):
        self.model = source_model
        self.model = self.model.cuda()
        self.adversary = attack_class(self.model, *args, **kwargs)
        self.loader = dataloader
        self.perturbed_dataset = []
        self.perturbed_dataset_length = 0
        self.max_instances=max_instances
        self.binary_classification = binary_classification
        self.targeted = False
    

    def generate_examples(self, force_attack = True):
        if (not self.perturbed_dataset) or force_attack:
            self.perturbed_dataset = []
            self.perturbed_dataset_length = min(max(self.max_instances,self.loader.batch_size),len(self.loader.dataset)) if self.max_instances>0 else len(self.loader.dataset)
            max_attacks = (self.perturbed_dataset_length+self.loader.batch_size-1)//self.loader.batch_size
            print("Generating %d adversarial examples"%self.perturbed_dataset_length)
            for i,(x,y) in enumerate(self.loader):                
                if self.binary_classification:
                    y = (y == 0).float().view(-1,1)
                advimg = self.adversary.perturb(x.cuda(),y.cuda())                
                self.perturbed_dataset.append((advimg,y))
                if i+1>=max_attacks:
                    break

    def eval(self, attacked_model = None, force_attack = False):
        self.generate_examples(force_attack = force_attack)
        confusion_matrix = None
        if not attacked_model:
            attacked_model = self.model
        attacked_model = attacked_model.cuda()
        
        correct = 0
        for x,y in self.perturbed_dataset:
            logits = attacked_model(x.cuda())
            
            if confusion_matrix is None:
                nclasses = logits.shape[1]
                confusion_matrix = np.zeros((nclasses, nclasses))
            
            pred = torch.argmax(logits, dim=1)
            correct += (pred.cpu() == y).sum()
            
            for t,p in zip(y, pred):
                t = t.int()
                confusion_matrix[t,p] += 1
        
        confusion_matrix /= np.sum(confusion_matrix, axis=1, keepdims=True)
        accuracy = correct.float()/self.perturbed_dataset_length
        
        return accuracy.item(), confusion_matrix

def extract_attack(args):
    if args.binary_classification:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9), reduction='sum')
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
    if args.attack =="fgsm":
        attack_class = FGSM
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps}
        print("Using FGSM attack with eps=%f"%args.eps)
    elif args.attack =="pgdinf":
        attack_class = LinfPGDAttack
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps,"nb_iter":args.nb_iter,"eps_iter":args.eps_iter}
        print("Using PGD attack with %d iterations of step %f on Linf ball of radius=%f"%(args.nb_iter,args.eps_iter,args.eps))
    elif args.attack =="pgdl2":
        attack_class = L2PGDAttack
        attack_kwargs = {"loss_fn":loss_fn,"eps":args.eps,"nb_iter":args.nb_iter,"eps_iter":args.eps_iter}
        print("Using PGD attack with %d iterations of step %f on L2 ball of radius=%f"%(args.nb_iter,args.eps_iter,args.eps))
    elif args.attack =="cwl2":
        attack_class = CarliniWagnerL2Attack
        attack_kwargs = {"loss_fn":None,"num_classes":10,"confidence":args.conf,"max_iterations":args.max_iters,"learning_rate":args.lr}
        print("Using Carlini&Wagner attack with %d iterations of step %f and confidence %f"%(args.max_iters,args.lr,args.conf))
    elif args.attack =="jsma":
        attack_class = JacobianSaliencyMapAttack
        attack_kwargs = {"loss_fn":None,"num_classes":10,"theta":args.eps,"gamma":args.gamma,}
        print("Using JSMA attack with %d theta %f and gamma %f"%(args.max_iters,args.eps,args.gamma))
    else:
        print("No known attack specified : test set will be used")
        attack_class = NullAdversary
        attack_kwargs={}

    return attack_class,attack_kwargs

def whitebox_attack(args):
    print("Using a white box attack")    
    model = NormalizationWrapper(torch.load(args.model_path))
    model.eval()
    test_loader = get_test_loader(args.dataset, batch_size=args.batch_size)   
    print("Model configuration")
    
    attack_class,attack_kwargs = extract_attack(args)
    
    # attacker = Attacker(model,test_loader, attack_class=attack_class, max_instances=args.max_instances, 
    #                     clip_min=0., clip_max=1., targeted=False, binary_classification=args.binary_classification, 
    #                     **attack_kwargs)
    # accuracy, confusion_matrix = attacker.eval()
    # print("Accuracy under attack : %f"%accuracy)
    # print('Confusion Matrix:')    
    # print(np.diag(confusion_matrix))

    attackers = [attack_class(model, **attack_kwargs) for i in range(args.nb_restarts)]
    if len(attackers) > 1:
        attacker = ChooseBestAttack(model, attackers, targeted=attackers[0].targeted)
    else:
        attacker = attackers[0]
    adv, label, pred, advpred = attack_whole_dataset(attacker, test_loader)
    print('clean accuracy:',get_accuracy(pred, label))
    print('robust accuracy:',get_accuracy(advpred, label))    

    outfile = os.path.join(os.path.dirname(args.model_path), 'advdata_%s_eps=%f.npy' % (args.attack, args.eps))
    torch.save({
        'data': adv,
        'preds': advpred,
        'labels': label
    }, outfile)


def transfer_attack(args):    
    model = NormalizationWrapper(torch.load(args.model_path))
    target_model = NormalizationWrapper(torch.load(args.target_model_path))
    model.eval()
    target_model.eval()
    if args.datafolder:
        model.args.datafolder = args.datafolder
    print("Source model configuration")
    print(model)
    print("Target model configuration")
    print(target_model)
    test_loader = get_test_loader(args.dataset, batch_size=128)
    
    attack_class,attack_kwargs = extract_attack(args)

    attacker = Attacker(model,test_loader, attack_class=attack_class, max_instances=args.max_instances, clip_min=0., clip_max=1., targeted=False, **attack_kwargs)

    accuracy = attacker.eval(attacked_model = target_model)
    print("Accuracy under attack : %f"%accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('model_path', type=str)
    parser.add_argument('--target_model_path', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datafolder', type=str, default='/home/mshah1/workhorse3/')
    parser.add_argument('--attack', type=str, default="none")
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--eps_iter', type=float, default=0.01)
    parser.add_argument('--conf', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_iters', type=float, default=100)
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()
    args.dataset = args.dataset.upper()

    if args.target_model_path:
        transfer_attack(args)
    else:
        whitebox_attack(args)
