import torch
from distillation import StudentModelWrapper
import os

baseline_dir = 'models/baseline'
distilled_dir = 'models/distilled'
model_names = [
    ('alexnet_cifar10','alexnet_cifar10',),
    ('alexnet_cifar10','alexnet_cifar10_allLayers',),
    ('alexnet_cifar10','alexnet_cifar10_allLayers_studentTargets',),
    ('vgg16_cifar10','vgg16_cifar10',),
    ('alexnet_caltech101_1C','alexnet_caltech101_1C',),
    ('alexnet_caltech101_1C','.alexnet_caltech101_1C_allLayers',),
    ('alexnet_caltech101_1C','alexnet_caltech101_allLayers_reverse',),
    ('alexnet_caltech101_1C','alexnet_caltech101_allLayers_reverse_lw75',),
    ('alexnet_caltech101_1C','alexnet_caltech101_allLayers_reverse_lw25',),
    ('vgg16_caltech101_1C','vgg16_caltech101_1C',),
    ('vgg16_caltech101_1C','vgg16_caltech101_1C_allLayers',),
    ('alexnet_caltech256_1C','alexnet_caltech256_1C',),
    ('alexnet_caltech256_1C','alexnet_caltech256_1C_allLayers',),
    ('vgg16_caltech256_1C','vgg16_caltech256_1C',),
    ('vgg16_caltech256_1C','vgg16_caltech256_1C_allLayers',),
]

for bmn,dmn in model_names:
    print(bmn,dmn)
    m = torch.load(os.path.join(baseline_dir, "%s.pt" % bmn))
    nparams_b = sum(p.numel() for p in m.parameters())

    m = torch.load(os.path.join(distilled_dir, "%s_distilled.pt" % dmn))
    nparams_d = sum(p.numel() for p in m.parameters())

    print(nparams_b, nparams_d, (nparams_d-nparams_b), 1-(nparams_b-nparams_d)/nparams_b)