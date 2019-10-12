import torch
from torch import nn

class ModelWrapper(object):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        self.bottleneck = None        

    def shrink_bottleneck(self, n=1):
        pass

class AlexNet(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, (5,5), (1,1), (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, (3,3), (1,1), (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(384, 256, (3,3), (1,1), (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 256, (3,3), (1,1), (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.bottleneck = nn.Linear(4096,4096)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        f = self.avgpool(self.features(x))
        f = f.view(f.shape[0], -1)
        return self.classifier(self.bottleneck(self.fc(f)))
    
    def shrink_bottleneck(self, n=1):
        outsize, insize = self.bottleneck.weight.shape
        new_outsize = max(1,outsize-n)
        self.bottleneck = nn.Linear(insize, new_outsize)        
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(new_outsize, 10)
        )

class AlexNetCIFAR10(AlexNet):
    def __init__(self, num_classes):
        super(AlexNetCIFAR10, self).__init__(num_classes)
        self.features = nn.Sequential(
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
        )        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

class PapernotCIFAR10(nn.Module, ModelWrapper):
    def __init__(self, num_classes):
        super(PapernotCIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(8192, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
        )
        self.bottleneck = nn.Linear(256, 256)
        self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(256, num_classes, bias=False), name='weight')
        )
    
    def forward(self, x):
        f = self.features(x).view(x.shape[0],-1)
        return self.classifier(self.bottleneck(self.fc(f)))
    
    def shrink_bottleneck(self, n=1):
        outsize, insize = self.bottleneck.weight.shape
        new_outsize = max(1,outsize-n)
        self.bottleneck = nn.Linear(insize, new_outsize)
        n_classes = list(self.classifier.modules())[-1].weight.shape[0]
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(new_outsize, n_classes, bias=False), name='weight')
        )

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

    