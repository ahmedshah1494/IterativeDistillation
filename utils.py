import numpy as np
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchvision
from torchvision import transforms

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

shrinkable_types = [nn.Linear, nn.Conv2d]
is_shrinkable = lambda l: 1 in [int(isinstance(l,t)) for t in shrinkable_types]

def change_layer_output(layer, new_size=None, factor=1, difference=0):
    if isinstance(layer, nn.Linear):
        outsize, insize = layer.weight.shape
        if new_size is None:
            new_size = int((outsize * factor) - difference)
        new_layer = nn.Linear(insize, new_size)
    elif isinstance(layer, nn.Conv2d):
        if new_size is None:
            new_size = int((layer.out_channels * factor) - difference)
        new_layer = nn.Conv2d(
            layer.in_channels,
            new_size,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.transposed,
            layer.output_padding,
            layer.groups,
            layer.padding_mode,
        )
    else:
        raise NotImplementedError('%s not supported for size changing' % str(type(layer)))

    return new_layer, new_size

def change_layer_input(layer, new_size):
    if isinstance(layer, nn.Linear):
        outsize, insize = layer.weight.shape            
        new_layer = nn.Linear(new_size, outsize)
    elif isinstance(layer, nn.Conv2d):            
        new_layer = nn.Conv2d(
            new_size,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.transposed,
            layer.output_padding,
            layer.groups,
            layer.padding_mode,
        )
    else:
        raise NotImplementedError('%s not supported for size changing' % str(type(layer)))

    return new_layer