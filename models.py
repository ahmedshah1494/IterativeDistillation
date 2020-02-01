import torch
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
import numpy as np
import torchvision
import utils
import types
import time
import sys

class ModelWrapper(object):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        self.layers = []
        self.device = None
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

    def shrink_layer(self, i, data, factor=1, difference=0):
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

class ModelWrapper2(ModelWrapper):
    def __init__(self):
        super(ModelWrapper2, self).__init__()
        self.layers = []
        self.device = None 
        self.args = None          

    # def estimate_z(self, Z_, zj):
    #     q,_ = torch.qr(Z_)
    #     zh = q.mm(q.transpose(0,1)).mm(zj)
    #     return zh

    # def score_neurons(self, Z):
    #     scores = []
    #     for j in range(Z.shape[1]):
    #         idxs = [k for k in range(Z.shape[1]) if j != k]
    #         Z_ = Z[:,idxs]
    #         zj = Z[:, [j]]
    #         zh = self.estimate_z(Z_, zj)
    #         error = torch.norm((zj-zh)).detach().cpu()
    #         scores.append(-error)
    #     return None, scores

    def score_neurons(self, Z, init_A = None):
        size = Z.shape[1]
        inv_eye = (1-torch.eye(size, device=self.device, requires_grad=False))
        if init_A is None:
            A = torch.rand((size, size), device=self.device)        
            A *= inv_eye
        else:
            A = init_A
        L = torch.rand((1,), device=self.device, requires_grad=True)

        A_optimizer = torch.optim.Adam([A, ], lr=1e-2)

        A.requires_grad = True
        L.requires_grad = True
                
        avg_error = torch.zeros((size,), device=self.device)
        criterion = lambda x,y: ((x-y)**2).sum(0).mean() + 0.0001*torch.abs(A).mean()
        prev_error = sys.maxsize
        for e in range(100):
            A_optimizer.zero_grad()
            error = criterion(Z, Z.mm(A))
            loss = error
            loss.backward()
            A.grad *= inv_eye
            A_optimizer.step()
            if error > prev_error:
                break
            else:
                prev_error = error
        A = A.detach()
        avg_error = torch.sqrt(((Z-Z.mm(A))**2).sum(0)).detach().cpu().numpy()
        scores = -1 * avg_error
        return A, scores
    
    def compute_prune_probability(self, i, data):
        trunc_model = self.layers[:i+1]
        Zs = []
        for batch in data:
            batch = batch.to(self.device)
            Z = trunc_model(batch)
            Zs.append(Z.detach())
        Z = torch.cat(Zs,dim=0)

        Z = Z.permute([j for j in range(len(Z.shape)) if j != 1]+[1])
        Z = Z.contiguous().view(-1, Z.shape[-1])        
        A, scores = self.score_neurons(Z)
        print('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        self.logger.info('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        return A, np.array(scores), Z

    def adjust_weights(self, W, A, J):
        J = sorted(J)
        d = W.shape[1]
        w_norm = init_w_norm = torch.norm(W,dim=1).mean()        
        for ji, j in enumerate(J):
            if w_norm / init_w_norm > 2:
                break
            print(ji, 'Aj = ', float((1-A[:,j]*A[j]).min()), float((1-A[:,j]*A[j]).max()))
            print(ji, 'W =', float(W.min()), float(W.max()))
            print('')

            j -= d-W.shape[1]        
            bJ = [i for i in range(W.shape[1]) if i != j]

            # method 1
            W += (A[:,[j]].mm(W.transpose(0,1)[[j]])).transpose(0,1)
            W = W[:,bJ]
            inv_eye = (1-torch.eye(A.shape[1], device=A.device))

            A += (A[:,[j]].mm(A[[j]]) * inv_eye)/(1-A[:,j]*A[j])
            A = A[bJ][:,bJ]

            if not torch.isfinite(A).any():
                print('A =',A)
                exit(0)

            if not torch.isfinite(W).any():
                print(ji, 'Aj = ', (1-A[:,j]*A[j]).max(), (1-A[:,j]*A[j]).min())
                print(ji, 'W =',W)
                exit(0)
            
            w_norm = torch.norm(W,dim=1).mean()            
            # W, A = W_, A_
        print('change in weight norm: %.4f -> %.4f' % (init_w_norm, w_norm))
        return W

    def remove_input_neurons(self, layer_idx, neuron_idxs, A=None):
        outsize, insize = utils.get_layer_input_output_size(self.layers[layer_idx])            
        retain_idxs = [i for i in range(insize) if i not in neuron_idxs]
        layer = self.layers[layer_idx]
        W = layer.weight.data
        if isinstance(layer, nn.Linear):
            if A is not None:
                W = self.adjust_weights(W, A, neuron_idxs)
            else:
                W = W[:, retain_idxs]
            new_layer = nn.Linear(W.shape[1], outsize)
            new_layer.weight.data = W            
        elif isinstance(layer, nn.BatchNorm2d):
            W = W[retain_idxs]
            b = layer.bias[retain_idxs]
            new_layer = nn.BatchNorm2d(W.shape[0])
            new_layer.weight.data = W
            new_layer.bias.data = b
        elif isinstance(layer, nn.Conv2d):            
            W = W[:, retain_idxs]
            new_layer = nn.Conv2d(W.shape[1], outsize,
                                layer.kernel_size,
                                layer.stride,
                                layer.padding,
                                layer.dilation)
            new_layer.weight.data = W
        else:
            raise NotImplementedError
        return new_layer

    def remove_output_neurons(self, layer_idx, neuron_idxs):
        outsize, insize = utils.get_layer_input_output_size(self.layers[layer_idx])            
        retain_idxs = [i for i in range(outsize) if i not in neuron_idxs]
        layer = self.layers[layer_idx]
        
        W = layer.weight.data
        W = W[retain_idxs]

        b = layer.bias.data
        b = b[retain_idxs]
        if isinstance(layer, nn.Linear):
            new_layer = nn.Linear(insize, W.shape[0])
            new_layer.weight.data = W
            new_layer.bias.data = b
        elif isinstance(layer, nn.Conv2d):
            new_layer = nn.Conv2d(insize, W.shape[0],
                                    layer.kernel_size,
                                    layer.stride,
                                    layer.padding,
                                    layer.dilation)
            new_layer.weight.data = W
            new_layer.bias.data = b
        else:
            raise NotImplementedError

        return new_layer

    def shrink_layer(self, i, data, factor=1, difference=0):
        if i == len(self.layers)-1 or i == -1:
            raise IndexError('Can not shrink output layer')
        outsize, _ = utils.get_layer_input_output_size(self.layers[i])
        new_size = int(outsize*factor - difference)
        A, salience_scores, Z = self.compute_prune_probability(i, data)
        prune_probs = utils.Softmax(salience_scores, 0, 50)
        # salience_scores = np.random.rand(outsize)
        # salience_scores /= np.sum(salience_scores)
        print(salience_scores.shape)        
        if salience_scores.shape[0] <= 1:
            return False
        pruned_neurons = np.random.choice(np.arange(len(salience_scores)), 
                                            (outsize - new_size), 
                                            replace=False, p=prune_probs)        
        print(salience_scores)
        print(prune_probs)
        print('mean error in removed neurons: ', np.mean(salience_scores[pruned_neurons]))
        self.logger.info('mean error in removed neurons: %f' % np.mean(salience_scores[pruned_neurons]))
        pruned_neurons = sorted(pruned_neurons)
        
        k = i+1
        while not utils.is_input_modifiable(self.layers[k]):
            k += 1
        
        _, old_insize = utils.get_layer_input_output_size(self.layers[k])
        init_w_norm = torch.norm(self.layers[k].weight.data, dim=1).mean()
        if self.args.readjust_weights:
            self.layers[k] = self.remove_input_neurons(k, pruned_neurons, A)
            # for ij,j in enumerate(pruned_neurons):
            #     _,insize = utils.get_layer_input_output_size(self.layers[k])
            #     print(ij, j, insize, A.shape, Z.shape)
            #     j = j-ij
            #     l = self.remove_input_neurons(k, [j], A)
                
            #     bj = [i for i in range(Z.shape[1]) if i != j]
            #     Z = Z[:, bj]
            #     A = A[bj][:, bj]
            #     A,_ = self.score_neurons(Z,A)

            #     w_norm = torch.norm(l.weight.data, dim=1).mean()      
            #     if w_norm/init_w_norm <= 2:
            #         self.layers[k] = l
            #     else:
            #         break
        else:
            self.layers[k] = self.remove_input_neurons(k, pruned_neurons)
        w_norm = torch.norm(self.layers[k].weight.data, dim=1).mean()
        print('change in weight norm: %.4f -> %.4f' % (init_w_norm, w_norm))

        _, new_insize = utils.get_layer_input_output_size(self.layers[k])
        
        self.layers[i] = self.remove_output_neurons(i, pruned_neurons[:old_insize-new_insize])
        outsize, insize = utils.get_layer_input_output_size(self.layers[i])
        return True

# class ModelWrapper2(ModelWrapper):
#     def __init__(self):
#         super(ModelWrapper2, self).__init__()

#         self.layers = []
    
#     def estimate_z(self, Z_, zj):
#         # return torch.mm(torch.mm(Z_,torch.pinverse(Z_)), zj)
#         q,_ = torch.qr(Z_)
#         zh = q.mm(q.transpose(0,1)).mm(zj)
#         return zh

#     def compute_saliency_scores(self, i, data):
#         trunc_model = self.layers[:i+1]
#         Z = trunc_model(data)
#         scores = []
#         weight_idxs = np.arange(Z.shape[1]) #np.random.choice(range(Z.shape[1]), Z.shape[0], replace=False)
#         for k,j in enumerate(weight_idxs):            
#             idxs = list(range(Z.shape[1]))
#             idxs.remove(j)
#             Z_ = Z[:, idxs]
#             zj = Z[:,[j]]
#             zh = self.estimate_z(Z_, zj)
#             error = -torch.norm(zj-zh).detach().cpu()
#             scores.append(float(error))
#         scores = softmax(scores)
#         sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)        
#         sorted_weight_idx = weight_idxs[sorted_idx]
#         return sorted_weight_idx
    
#     def shrink_input(self, layer_idx, neuron_idxs):
#         l = self.layers[layer_idx]
#         if isinstance(l, nn.Linear):
#             W = l.weight
#             outsize, insize = W.shape            
#             idxs = list(range(W.shape[1]))
#             for ni in neuron_idxs:
#                 idxs.remove(ni)
#             W = W[:, idxs]
#             outsize, insize = W.shape
#             new_layer = nn.Linear(insize, outsize)
#             new_layer.weight.data = W
#         else:
#             raise NotImplementedError
#         return new_layer
    
#     def shrink_output(self, layer_idx, neuron_idxs):
#         l = self.layers[layer_idx]
#         if isinstance(l, nn.Linear):
#             W = l.weight
#             outsize, insize = W.shape
#             idxs = list(range(W.shape[0]))            
#             for ni in neuron_idxs:
#                 idxs.remove(ni)
#             W = W[idxs, :]
#             outsize, insize = W.shape
#             new_layer = nn.Linear(insize, outsize)                        
#             new_layer.weight.data = W

#             idxs = list(range(l.bias.shape[0]))
#             for ni in neuron_idxs:
#                 idxs.remove(ni)
#             new_layer.bias.data = l.bias[idxs]
#         else:
#             raise NotImplementedError
#         return new_layer

#     def shrink_layer(self, i, data, factor=1, difference=0):
#         if i == len(self.layers)-1 or i == -1:
#             raise IndexError('Can not shrink output layer')
#         outsize, _ = utils.get_layer_input_output_size(self.layers[i])
#         new_size = int(outsize*factor - difference)
#         print(outsize, new_size)
#         while outsize > new_size:
#             t0 = time.time()
#             sorted_idx = self.compute_saliency_scores(i, data)
#             print('time for saliency computation', time.time() - t0)
#             prune_idxs = sorted_idx[:(outsize - new_size)]
#             self.layers[i] = self.shrink_output(i, prune_idxs)
#             k = i+1
#             while not utils.is_input_modifiable(self.layers[k]):
#                 k += 1
#             self.layers[k] = self.shrink_input(k, prune_idxs)
#             outsize, _ = utils.get_layer_input_output_size(self.layers[i])
#             print(outsize)
#         return True

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
    