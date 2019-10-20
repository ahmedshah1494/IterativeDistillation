import torch
from torch import nn
import numpy as np
import torchvision
import utils
import types

class ModelWrapper(object):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        self.layers = []
        
    def get_shrinkable_layers(self, non_shrinkables=[]):
        shrinkable_layers = [i for i,l in enumerate(self.layers[:-1]) if utils.is_output_modifiable(l) and i not in non_shrinkables]
        return shrinkable_layers
    
    def is_last_conv(self,i):
        if not isinstance(self.layers[i], nn.Conv2d):
            return False
            
        for j in range(i+1, len(self.layers)):
            if isinstance(self.layers[j], nn.Conv2d):
                return False
        return True

    def replace_layer(self, i, new_layer):
        layers = list(self.layers.children())
        layers[i] = new_layer
        self.__delattr__('layers')
        self.layers = nn.Sequential(*layers)

    def shrink_layer(self, i, factor=1, difference=0):
        if i == len(self.layers)-1 or i == -1:
            raise IndexError('Can not shrink output layer')
        out_size, _ = utils.get_layer_input_output_size(self.layers[i])         
        new_layer , new_size = utils.change_layer_output(self.layers[i], factor=factor, difference=difference)
        if new_layer is None and self.is_last_conv(i):
            return False
        # self.layers[i] = new_layer
        self.replace_layer(i, new_layer)
        if self.layers[i] is None:
            while not utils.is_weighted(self.layers[i]):
                print('deleting',self.layers[i])
                self.layers.__delitem__(i)
            # self.layers[i] = utils.change_layer_input(self.layers[i], new_size)
            new_layer = utils.change_layer_input(self.layers[i], new_size)
            self.replace_layer(i, new_layer)
            return False
        else:
            while i < len(self.layers)-1:
                i += 1
                if utils.is_input_modifiable(self.layers[i]):
                    _,in_size = utils.get_layer_input_output_size(self.layers[i])
                    scale = in_size // out_size
                    # self.layers[i] = utils.change_layer_input(self.layers[i], new_size*scale)
                    new_layer = utils.change_layer_input(self.layers[i], new_size*scale)
                    self.replace_layer(i, new_layer)
                    if not isinstance(self.layers[i], nn.BatchNorm2d):
                        break
            return True

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)

class AlexNet(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class AlexNetCIFAR(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(AlexNetCIFAR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.layers(x)

class PapernotCIFAR10(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(PapernotCIFAR10, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(256, num_classes, bias=False), name='weight')
        )
    
    def forward(self, x):
        return self.layers(x)

class PapernotCIFAR10_2(PapernotCIFAR10):
    def __init__(self, num_classes):
        super(PapernotCIFAR10_2, self).__init__(num_classes)

        self.bottleneck = nn.Linear(8192, 256)
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                self.bottleneck,
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(256, num_classes, bias=False), name='weight')
        )
    
    def forward(self, x):
        f = self.features(x).view(x.shape[0],-1)
        return self.classifier(f)
    
    def shrink_bottleneck(self, n=1):
        outsize, insize = self.bottleneck.weight.shape
        new_outsize = max(1,outsize-n)
        self.bottleneck = nn.Linear(insize, new_outsize)
        n_classes = list(self.classifier.modules())[-1].weight.shape[0]
        self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                self.bottleneck,
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(new_outsize, 256),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(256, n_classes, bias=False), name='weight')
        )

class VGG16(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
    
        self.layers = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        return self.layers(x)

class VGG16CIFAR(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(VGG16CIFAR, self).__init__()
    
        self.layers = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.Conv2d(512,512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.layers(x)

def modified_fwd(self, x):
    bs = x.shape[0]
    if len(x.shape) == 5:        
        x = utils.reshape_multi_crop_tensor(x)
    out = self.old_forward(x)
    out = out.view(bs, -1, out.shape[1])
    out = torch.mean(out, dim=1)
    return out

def get_classifier(indim, num_classes, depth):
    width = 2**(int(np.log2(num_classes)) + 2)
    layers = []
    for i in range(depth):
        if i > 0:
            indim = width
        if i == depth-1:
            outdim = num_classes
        else:
            outdim = width
        layers.append(nn.Linear(indim, outdim))
        if i < depth-1:
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout())
    classifier = nn.Sequential(*layers)
    return classifier

def resnet18TinyImageNet(num_classes, pretrained=False, feature_extraction=False, **kwargs):
    m = torchvision.models.resnet18(pretrained=pretrained)
    m.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, 200)
    m.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    m.maxpool = nn.Sequential()    
    for param in m.fc.parameters():
        param.requires_grad = True
    return m

def resnet18(num_classes, pretrained=False, feature_extraction=False, classifier_depth=1):
    m = torchvision.models.resnet18(pretrained=pretrained)
    if pretrained and feature_extraction:
        for param in m.parameters():
            param.requires_grad = False
    num_ftrs = m.classifier[-1].in_features
    m.classifier[-1] = get_classifier(num_ftrs, num_classes, classifier_depth)    
    for param in m.fc.parameters():
        param.requires_grad = True      
    return m

def vgg16(num_classes, pretrained=False, feature_extraction=False, classifier_depth=1):
    m = torchvision.models.vgg16_bn(pretrained=pretrained)
    if pretrained and feature_extraction:
        for param in m.parameters():
            param.requires_grad = False
    # m.classifier[-1], _ = utils.change_layer_output(m.classifier[-1], num_classes)
    num_ftrs = m.classifier[-1].in_features
    m.classifier[-1] = get_classifier(num_ftrs, num_classes, classifier_depth)
    for param in m.classifier[-1].parameters():
        param.requires_grad = True
    return m

def vgg16CIFAR(num_classes, pretrained=False, feature_extraction=False, **kwargs):
    return VGG16CIFAR(num_classes)

def alexnet(num_classes, pretrained=False, feature_extraction=False, classifier_depth=1):
    m = torchvision.models.alexnet(pretrained=pretrained)
    if pretrained and feature_extraction:
        for param in m.parameters():
            param.requires_grad = False
    num_ftrs = m.classifier[-1].in_features
    m.classifier[-1] = get_classifier(num_ftrs, num_classes, classifier_depth)
    for param in m.classifier[-1].parameters():
        param.requires_grad = True
    return m

def alexnetCIFAR(num_classes, pretrained=False, feature_extraction=False, **kwargs):
    return AlexNetCIFAR(num_classes)

def papernot(num_classes, pretrained=False, feature_extraction=False, **kwargs):
    m = PapernotCIFAR10(num_classes)
    return m

def setup_feature_extraction_model(model, model_type, num_classes, classifier_depth=1):
    for p in model.parameters():
        p.requires_grad = False
    if model_type == 'vgg16' or model_type =='AlexNet':        
        num_ftrs = model.classifier[-1][-1].in_features
        model.classifier[-1][-1] = get_classifier(num_ftrs, num_classes, classifier_depth)
        for param in model.classifier[-1][-1].parameters():
            param.requires_grad = True
    else:
        old_classifier = model.layers[-1]
        model.layers[-1] = get_classifier(old_classifier.in_features, num_classes, classifier_depth)
        for p in model.layers[-1].parameters():
            p.requires_grad = True
    