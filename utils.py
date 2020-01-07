import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
from scipy.special import softmax

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

compute_correct = lambda out,true: float(torch.sum((torch.argmax(out, 1) - true) == 0))

def label_counts(loader, nclasses):
    label_counts = [0]*nclasses
    for _,y in loader:
        for i in y:
            label_counts[i] += 1
    label_counts = np.array(label_counts)
    return label_counts

def compute_correct_per_class(out, true, nclasses):
    preds = torch.argmax(out,1)
    correct = np.zeros(nclasses)
    labels = set(list(true))
    for l in labels:        
        correct[l] = torch.sum(preds[true == l] == l)
    return correct

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

def reshape_multi_crop_tensor(x):
    return x.view(-1, *(x.shape[2:]))

def loss_wrapper(margin):
    if margin == 0:
        return nn.functional.cross_entropy
    else:
        return lambda x,y: nn.functional.multi_margin_loss(x/torch.sum(x,1,keepdim=True), y, margin=margin)

def get_common_transform(training=True):
    if training:
        transform_list = [        
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(30), 
            torchvision.transforms.RandomGrayscale(p=0.1),
            ]
    else:
        transform_list = []
    transform_list += [torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )]
    transform = torchvision.transforms.Compose(transform_list)
    return transform

normalize_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
rescale_and_crop_transform = lambda f: transforms.Compose([
    transforms.Resize(int(384*f)),
    transforms.RandomCrop(224, pad_if_needed=True)
])
multi_scale_transform = transforms.Compose([
    transforms.Lambda(lambda img: [rescale_and_crop_transform(f)(img) for f in [0.67,1,1.33]]),
    transforms.Lambda(lambda crops: crops + [transforms.functional.hflip(c) for c in crops]),
    transforms.Lambda(lambda crops: torch.stack([normalize_transform(transforms.ToTensor()(crop)) for crop in crops])),
])        

def channel_first_transform(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.from_numpy(np.array(x)).float()
    return x.transpose(2,1).transpose(0,1)

def reinitialize_model(model):
    for param in model.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.xavier_uniform_(param.data)
        else:
            torch.nn.init.constant_(param.data,0)

modifiable_output = [nn.Linear, nn.Conv2d]
weighted_layers = [nn.Linear, nn.Conv2d]
modifiable_input = [nn.Linear, nn.Conv2d, nn.BatchNorm2d]
is_output_modifiable = lambda l: 1 in [int(isinstance(l,t)) for t in modifiable_output]
is_weighted = lambda l: 1 in [int(isinstance(l,t)) for t in weighted_layers]
is_input_modifiable = lambda l: 1 in [int(isinstance(l,t)) for t in modifiable_input]

def change_layer_output(layer, new_size=None, factor=1, difference=0):    
    if isinstance(layer, nn.Linear):
        outsize, insize = layer.weight.shape
        if new_size is None:
            new_size = int((outsize * factor) - difference)
        if new_size == outsize:
            return layer, new_size
        if new_size < 1:
            return None,insize
        new_layer = nn.Linear(insize, new_size)    
    elif isinstance(layer, nn.Conv2d):
        if new_size is None:
            new_size = int((layer.out_channels * factor) - difference)
        if new_size == layer.out_channels:
            return layer, new_size
        if new_size < 1:
            return None,layer.in_channels
        new_layer = nn.Conv2d(
            layer.in_channels,
            new_size,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
        )
    else:
        raise NotImplementedError('%s not supported for output size modification' % str(type(layer)))

    return new_layer, new_size

def change_layer_input(layer, new_size):
    if new_size == 0:
        return None
    if isinstance(layer, nn.Linear):
        outsize, insize = layer.weight.shape
        if new_size == insize:
            return layer        
        new_layer = nn.Linear(new_size, outsize)
    elif isinstance(layer, nn.BatchNorm2d):
        size = layer.num_features
        if new_size == size:
            return layer
        new_layer = nn.BatchNorm2d(new_size)
    elif isinstance(layer, nn.Conv2d):
        if new_size == layer.in_channels:
            return layer
        new_layer = nn.Conv2d(
            new_size,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation
        )
    else:
        raise NotImplementedError('%s not supported for input size modification' % str(type(layer)))

    return new_layer

def Softmax(data, axis, T):
    return softmax(data/T, axis=axis)
    
def get_layer_input_output_size(layer):
    if isinstance(layer, nn.Linear):
        return layer.weight.shape
    elif isinstance(layer, nn.BatchNorm2d):
        return layer.num_features, layer.num_features
    elif isinstance(layer, nn.Conv2d):
        return layer.out_channels, layer.in_channels
    else:
        raise NotImplementedError('%s not supported' % str(type(layer)))

def get_datasets(args):
    common_transform = get_common_transform()
    test_transform = get_common_transform(training=False)
    if args.dataset == 'cifar10':        
        train_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', 
                    transform=common_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10('/home/mshah1/workhorse3/', train=False,
                    transform=test_transform, download=True)        
        nclasses = 10
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100('/home/mshah1/workhorse3/', 
                    transform=common_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100('/home/mshah1/workhorse3/', train=False,
                    transform=test_transform, download=True)        
        nclasses = 100
    elif 'caltech' in args.dataset:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])  # Imagenet standards
        ])

        test_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if args.dataset == 'caltech101':
            ntrain_files = 30
            nclasses = 102
        if args.dataset == 'caltech256':
            ntrain_files = 60
            nclasses = 257

        def is_valid_file(fn):
            return os.path.basename(fn).split('.')[-1] == 'jpg'
        def is_train_file(fn):
            return is_valid_file(fn) and int(os.path.basename(fn).split('.')[0].split('_')[-1]) <= ntrain_files
        def is_test_file(fn):
            return is_valid_file(fn) and int(os.path.basename(fn).split('.')[0].split('_')[-1]) > ntrain_files

        train_dataset = torchvision.datasets.ImageFolder('/home/mshah1/workhorse3/%s/' % args.dataset, 
                                                            transform=train_transform,
                                                            is_valid_file= is_train_file)        
        test_dataset = torchvision.datasets.ImageFolder('/home/mshah1/workhorse3/%s/' % args.dataset, 
                                                            transform=test_transform,
                                                            is_valid_file= is_test_file)
        print(train_dataset[0][0].shape)
        
    elif args.dataset == 'tiny_imagenet':
        train_transform = transforms.Compose([
            # transforms.Resize(224, Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            common_transform
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(224, Image.BICUBIC),
            test_transform
        ])
        train_dataset = torchvision.datasets.ImageFolder('/home/mshah1/workhorse3/tiny-imagenet-200/train/', 
                                                            transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder('/home/mshah1/workhorse3/tiny-imagenet-200/val/', 
                                                            transform=test_transform)
        print(train_dataset[0][0].shape)
        nclasses = 200
    else:
        raise NotImplementedError
    return train_dataset, test_dataset, nclasses