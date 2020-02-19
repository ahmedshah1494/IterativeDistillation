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

        regularizer_weight = 1e-4
        A_optimizer = torch.optim.Adam([A, ], lr=1e-3, weight_decay=regularizer_weight)

        A.requires_grad = True
                
        avg_error = torch.zeros((size,), device=self.device)        
        criterion = lambda x,y: torch.norm(x-y, p=2, dim=0).mean() #+ regularizer_weight * torch.norm(A, p=1, dim=0).mean()
        prev_error = sys.maxsize

        dataset = TensorDataset(Z)
        loader = DataLoader(Z, batch_size=128, shuffle=True)
        patience = 5
        bad_iters = 0
        for e in range(500):
            avg_loss = 0
            for bi,z in enumerate(loader):
                z = z.cuda()
                A_optimizer.zero_grad()
                error = criterion(z, z.mm(A))
                loss = error
                loss.backward()
                A.grad *= inv_eye
                A_optimizer.step()

                avg_loss = (bi*avg_loss + loss)/(bi+1)
            print(e+1, float(avg_loss), float(torch.norm(A, p=1, dim=0).mean()), bad_iters)
            if avg_loss > prev_error:
                if bad_iters >= patience -1:
                    break
                else:
                    bad_iters += 1
            else:
                bad_iters = 0
                prev_error = avg_loss
        
        avg_error = 0
        for bi,z in enumerate(loader):
            z = z.cuda()
            avg_error += torch.norm(z-z.mm(A), p=2, dim=0).detach().cpu().numpy()
        avg_error /= A.shape[0]

        A = A.detach()
        hist, bins = np.histogram(A.cpu().numpy().reshape(-1), bins=10)
        print('A_hist:', list(zip(hist.tolist(), bins.tolist())))
        hist, bins = np.histogram(avg_error.reshape(-1), bins=10)
        print('error_hist:', list(zip(hist.tolist(), bins.tolist())))

        A.to(self.device)
        # avg_error = torch.sqrt(((Z-Z.mm(A))**2).sum(0)).detach().cpu().numpy()
        # avg_error /= A.shape[0]
        scores = -1 * avg_error
        return A, scores
    
    def compute_prune_probability(self, i, data):
        trunc_model = self.layers[:i+1]
        Zs = []
        for batch in data:
            batch = batch.to(self.device)
            Z = trunc_model(batch)
            Zs.append(Z.detach().cpu())
        Z = torch.cat(Zs,dim=0)

        Z = Z.permute([j for j in range(len(Z.shape)) if j != 1]+[1])
        Z = Z.contiguous().view(-1, Z.shape[-1])

        ones = torch.ones((Z.shape[0],1), device=Z.device)
        Z = torch.cat((Z,ones), dim=1)
        
        A, scores = self.score_neurons(Z)
        scores = scores[:-1]

        Z = Z.detach().cpu().numpy()

        print('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        self.logger.info('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        return A, np.array(scores), np.mean(np.abs(Z), axis=0)

    def adjust_weights(self, W, A, J, mean_Z):
        J = sorted(J)
        d = W.shape[1]
        w_norm = init_w_norm = torch.norm(W,dim=1).mean()        
        for ji, j in enumerate(J):            
            # print(ji, 'Aj = ', float((1-A[:,j]*A[j]).min()), float((1-A[:,j]*A[j]).max()))
            # print(ji, 'W =', float(W.min()), float(W.max()))
            # print('')

            j -= d-W.shape[1]
            print(j, A.shape)
            bJ = [i for i in range(W.shape[1]) if i != j]

            # method 1
            w_update = (A[:,[j]].mm(W.transpose(0,1)[[j]])).transpose(0,1)
            update_norm = torch.norm(w_update, p=2, dim=1).mean()
            print('update_norm:', update_norm)
            
            if update_norm > 1 and update_norm > w_norm:
                print('update norm too large, breaking...')
                break
            
            W += w_update
            W = W[:,bJ]
            inv_eye = (1-torch.eye(A.shape[1], device=A.device))            

            future_error = (torch.abs(A[:, j]) * mean_Z[j]).mean()
            print('future error: %.4f' % future_error)
            if future_error > 1.0:
                break

            A += (A[:,[j]].mm(A[[j]]) * inv_eye)/(1-A[:,j]*A[j])
            A = A[bJ][:,bJ]

            # M = torch.ones(A.shape, device=A.device)
            # M[:, j] *= 0 
            # I = torch.eye(A.shape[1], device=A.device)
            # U = I*M + (1-M).transpose(0,1) * A * M
            # AbJ = (1-M).transpose(0,1) * A * (1-M)
            # V = I - AbJ
            # T = torch.inverse(V).mm(U)
            # print((U*inv_eye).nonzero())
            # print((AbJ*inv_eye).nonzero())
            # print((V*inv_eye).nonzero())
            # A = (1/(1-torch.diag(T))).view(-1,1)*(T*(1-I))
            # A = A[bJ][:, bJ]
            # A = T.mm(A)[bJ][:, bJ]

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

    def remove_input_neurons(self, layer_idx, neuron_idxs, A=None, mean_Z=None):
        outsize, insize = utils.get_layer_input_output_size(self.layers[layer_idx])            
        retain_idxs = [i for i in range(insize) if i not in neuron_idxs]
        layer = self.layers[layer_idx]
        if len(retain_idxs) == 0:
            return layer
        W = layer.weight.data
        b = layer.bias.data
        if isinstance(layer, nn.Linear):
            if A is not None:
                W = torch.cat((W, b.view(-1,1)), dim=1)
                W = self.adjust_weights(W, A, neuron_idxs, mean_Z)
                W, b = W[:,:-1], W[:,-1]
            else:
                W = W[:, retain_idxs]
            new_layer = nn.Linear(W.shape[1], outsize)
            new_layer.weight.data = W
            new_layer.bias.data = b
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
            new_layer.bias.data = b
        else:
            raise NotImplementedError
        return new_layer

    def remove_output_neurons(self, layer_idx, neuron_idxs):
        outsize, insize = utils.get_layer_input_output_size(self.layers[layer_idx])            
        retain_idxs = [i for i in range(outsize) if i not in neuron_idxs]
        layer = self.layers[layer_idx]

        if len(retain_idxs) == 0:
            return layer
        
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
        if self.args.readjust_weights:
            A, salience_scores, mean_Z = self.compute_prune_probability(i, data)
        else:
            _, salience_scores, _ = self.compute_prune_probability(i, data)


        prune_probs = utils.Softmax(salience_scores, 0, 1)

        # salience_scores = np.random.rand(outsize)
        # salience_scores /= np.sum(salience_scores)
        print(salience_scores.shape)        
        if salience_scores.shape[0] <= 1:
            return False
        # pruned_neurons = np.random.choice(np.arange(len(salience_scores)), 
        #                                     (outsize - new_size), 
        #                                     replace=False, p=prune_probs)  
        pruned_neurons = sorted(range(len(salience_scores)), 
                                    key=lambda k: salience_scores[k],
                                    reverse=True)[:(outsize - new_size)]        
        # print(salience_scores)
        # print(salience_scores[pruned_neurons])
        print('mean error in removed neurons: ', np.mean(salience_scores[pruned_neurons]))
        self.logger.info('mean error in removed neurons: %f' % np.mean(salience_scores[pruned_neurons]))
        pruned_neurons = sorted(pruned_neurons)
        
        k = i+1
        while k < len(self.layers):            
            if utils.is_input_modifiable(self.layers[k]):
                _, old_insize = utils.get_layer_input_output_size(self.layers[k])
                
                if self.is_last_conv(i):
                    num_last_conv_filts = self.layers[i].weight.data.shape[0]            
                    idxs = np.arange(old_insize).reshape(num_last_conv_filts, -1)
                    pruned_input_neurons_ = pruned_input_neurons = idxs[pruned_neurons].flatten()                    
                else:
                    pruned_input_neurons_ = pruned_input_neurons = pruned_neurons
                
                if isinstance(self.layers[k], nn.BatchNorm2d):
                    pruned_input_neurons = pruned_neurons

                init_w_norm = torch.norm(self.layers[k].weight.data, dim=-1).mean()
                if self.args.readjust_weights:                    
                    self.layers[k]= self.remove_input_neurons(k, pruned_neurons, A, mean_Z)                        
                else:
                    self.layers[k] = self.remove_input_neurons(k, pruned_input_neurons)
                pruned_input_neurons = pruned_input_neurons_

                w_norm = torch.norm(self.layers[k].weight.data, dim=-1).mean()
                print('change in weight norm: %.4f -> %.4f' % (init_w_norm, w_norm))

                if not isinstance(self.layers[k], nn.BatchNorm2d):
                    break
            k += 1
        _, new_insize = utils.get_layer_input_output_size(self.layers[k])
        
        if self.args.readjust_weights:
            self.layers[i] = self.remove_output_neurons(i, pruned_neurons[:old_insize-new_insize])
        else:
            self.layers[i] = self.remove_output_neurons(i, pruned_neurons)            
        outsize, insize = utils.get_layer_input_output_size(self.layers[i])
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
    