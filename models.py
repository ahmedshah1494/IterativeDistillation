import torch
from torch import nn

shrinkable_types = [nn.Linear, nn.Conv2d]
is_shrinkable = lambda l: 1 in [int(isinstance(l,t)) for t in shrinkable_types]

class ModelWrapper(object):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        self.layers = []
    def get_shrinkable_layers(self):
        shrinkable_layers = [i for i,l in enumerate(self.layers[:-1]) if is_shrinkable(l)]
        return shrinkable_layers

    def shrink_layer(self, i, factor=1, difference=0):
        if i == len(self.layers)-1 or i == -1:
            raise IndexError('Can not shrink output layer')        
        self.layers[i], new_size = self.change_layer_output(self.layers[i], factor, difference)
        i += 1
        while not is_shrinkable(self.layers[i]):
            i += 1
        self.layers[i] = self.change_layer_input(self.layers[i], new_size)
    
    @staticmethod
    def change_layer_output(layer, factor=1, difference=0):
        if isinstance(layer, nn.Linear):
            outsize, insize = layer.weight.shape
            new_size = int((outsize * factor) - difference)
            new_layer = nn.Linear(insize, new_size)
        elif isinstance(layer, nn.Conv2d):
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
    
    @staticmethod
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

    