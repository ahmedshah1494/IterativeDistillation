import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchvision

class AlphaCrossEntropyLoss(nn.Module):
    """docstring for alpha_softmax"""
    def __init__(self, alpha=1.0):
        super(AlphaCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def scale(self, X, y_true):
        idxs = [range(y_true.shape[0]), list(y_true)]
        alphav = torch.ones(X.shape, device=X.device)
        alphav[range(alphav.shape[0]), y_true] = self.alpha
        scaled = X * alphav
        return scaled       

    def forward(self, input, target):
        scaled = self.scale(input, target)
        return nn.CrossEntropyLoss()(scaled, target)

def curry(new_func, func_seq):
    return lambda x: new_func(func_seq(x))

compute_correct = lambda out,true: torch.sum((torch.argmax(out, 1) - true) == 0)

def evaluate(model, dataset, device):    
    loader = DataLoader(dataset, 1024, shuffle=False)
    val_correct = 0
    for i,batch in enumerate(loader):
        model.eval()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        val_correct += float(compute_correct(out,y))
    val_acc = val_correct / len(dataset) 
    return val_acc

def load_dataset(dataset, augment=False):
    if augment:
        transform = torchvision.transforms.RandomAffine(30)
    else:
        transform = lambda x:x
    transform = curry(lambda x: torch.from_numpy(np.array(x)).float(), transform)
    if dataset == 'CIFAR10':
        transform = curry(lambda x: x.transpose(2,1).transpose(0,1), transform)
        train_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', 
                    transform=transform, download=True)
        val_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', train=False,
                    transform=transform, download=True)
        input_shape = (-1,*(train_dataset[0][0].shape))
        nclasses = 10
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, input_shape, nclasses

def attack(attack_class, classifier, inputs, true_targets, epsilon):        
    adv_crafter = attack_class(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=inputs)    
    return x_test_adv