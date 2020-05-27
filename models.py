import torch
from torch.utils.data import DataLoader, TensorDataset

from torch import nn
import numpy as np
import torchvision
import utils
import types
import time
import sys
from wide_resnet import Wide_ResNet

class WideResNetEncoder(nn.Module):
    def __init__(self, ):
        super(WideResNetEncoder, self).__init__()
        self.encoder = Wide_ResNet(28, 10, 0.3, 10)    
        delattr(self.encoder, 'linear')
        
    def forward(self, x):
        out = self.encoder.conv1(x)
        out = self.encoder.layer1(out)
        out = self.encoder.layer2(out)
        out = self.encoder.layer3(out)
        out = torch.relu(self.encoder.bn1(out))
        out = nn.functional.avg_pool2d(out, 8)
        return out

class CIFAR10Classifier(nn.Module):
    def __init__(self, model_name):
        super(CIFAR10Classifier, self).__init__()

        fe_dict = {            
            'wide_resnet': lambda: WideResNetEncoder()
        }
        self.name = model_name+"Classifier"
        self.fe = fe_dict[model_name]()
        sample_output = self.fe(torch.rand((2,3,32,32)))
        sample_output = sample_output.view(sample_output.shape[0], -1)        
        self.classifier = nn.Linear(sample_output.shape[1], 10)
    
    def forward(self, x):
        x = utils.normalize_image_tensor(x)
        f = self.fe(x)
        f = f.view(f.shape[0], -1)
        logits = self.classifier(f)
        return logits

class ModelWrapper(object):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        self.layers = []
        self.device = None
    
    def reset(self):
        pass

    def get_Y(self, data, i):
        i += 1
        while i < len(self.layers) and not utils.is_weighted(self.layers[i]):
            i += 1
        trunc_model = self.layers[:i+1]
        trunc_model = trunc_model.eval()

        print(trunc_model)

        Ys = []
        for batch in data:
            x = batch[0]
            x = x.to(self.device)
            Y = trunc_model(x)
            Ys.append(Y.detach().cpu())
        Y = torch.cat(Ys,dim=0).numpy()

        return Y
    
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
    # shrinking_state = {
    #     'layer_idx': None,
    #     'A1': None,
    #     'A2': None
    # }
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
    # def update_shrinking_state(self, layer_idx, A1, A2):
    #     self.shrinking_state = {
    #         'layer_idx': layer_idx,
    #         'A1': A1,
    #         'A2': A2
    #     }    

    def score_neurons(self, Z, init_A = None, normalize=True):
        size = Z.shape[1]
        inv_eye = (1-torch.eye(size, device=self.device, requires_grad=False))        
        if init_A is None:
            A = torch.rand((size, size), device=self.device)        
            A *= inv_eye
        else:
            A = init_A.to(self.device)
        A.requires_grad = True
        regularizer_weight = 1e-4
        A_optimizer = torch.optim.Adam([A, ], lr=self.args.predictive_pruning_lr, weight_decay=regularizer_weight)
        A_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(A_optimizer, factor=0.5, patience=3, verbose=True)
                
        avg_error = torch.zeros((size,), device=self.device)        
        criterion = lambda x,y: torch.norm(x-y, p=2, dim=0).mean() #+ regularizer_weight * torch.norm(A, p=1, dim=0).mean()
        prev_error = sys.maxsize

        dataset = TensorDataset(Z)
        loader = DataLoader(Z, batch_size=self.args.predictive_pruning_batch_size, shuffle=True)
        patience = 10
        bad_iters = 0
        for e in range(self.args.predictive_pruning_iters):
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
            A_scheduler.step(avg_loss)
            print('%d/%d' % (e+1,self.args.predictive_pruning_iters), float(avg_loss), float(torch.norm(A, p=1, dim=0).mean()), bad_iters)
            if A_optimizer.param_groups[0]['lr'] <= 1e-6:
                break
            if np.isclose(float(avg_loss), float(prev_error)) or avg_loss > prev_error:
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
        if normalize:
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
    
    def compute_neuron_derivatives(self, Z, Y, model, normalize=True):
        dataset = TensorDataset(Z, Y)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        z_grad = np.zeros((Z.shape[1],))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        criterion = nn.CrossEntropyLoss()
        for bi,(z,y) in enumerate(loader):            
            z = z.to(self.device)
            z.requires_grad = True
            optimizer.zero_grad()
            logits = model(z)
            if self.args.scale_by_grad == 'output':
                logits.backward(torch.ones(logits.shape).to(z.device))                
            else:
                y = y.to(self.device)
                loss = criterion(logits, y)
                loss.backward()
            _z_grad = z.grad
            _z_grad = _z_grad.squeeze(0).view(z.shape[1],-1).sum(1).detach().cpu().numpy()            
            z_grad += _z_grad
        z_grad = np.abs(z_grad)
        if normalize:
            z_grad /= z_grad.sum()
        print(z_grad.shape)
        print(z_grad.max(), z_grad.mean(), z_grad.min())
        return z_grad
        
    def compute_prune_probability(self, i, data, flatten_conv_map=False, init_A1=None, init_A2=None, normalize=True):        
        k = i
        while i+1 < len(self.layers) and not utils.is_weighted(self.layers[i+1]) and not isinstance(self.layers[i+1], Flatten) and not isinstance(self.layers[i+1], nn.Dropout):
            i += 1
        trunc_model = self.layers[:i+1]
        trunc_model = trunc_model.eval()

        print('---------------------truncated_layers-----------------------')
        print(self.layers[k:i+1])
        print('------------------------------------------------------------')
        Zs = []
        Ys = []
        for x,y,_ in data:
            x = x.to(self.device)
            Z = trunc_model(x)
            Zs.append(Z.detach().cpu())
            Ys.append(y.detach().cpu())
        Z = torch.cat(Zs,dim=0)
        Y = torch.cat(Ys,dim=0)

        if self.args.scale_by_grad != 'none':
            scale = self.compute_neuron_derivatives(Z, Y, self.layers[i+1:], normalize=normalize)
        else:
            scale = 1

        Z = Z.permute([j for j in range(len(Z.shape)) if j != 1]+[1])
        if flatten_conv_map:
            Z = Z.contiguous().view(Z.shape[0], -1)
            init_A = init_A2
        else:
            Z = Z.contiguous().view(-1, Z.shape[-1])
            init_A = init_A1

        ones = torch.ones((Z.shape[0],1), device=Z.device)
        Z = torch.cat((Z,ones), dim=1)
        print('Z.shape =', Z.shape)
        A, scores = self.score_neurons(Z, init_A=init_A, normalize=normalize)        
        scores = scores[:-1]
        scores = scores*scale

        Z = Z.detach().cpu().numpy()
        torch.cuda.ipc_collect()

        print('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        # self.logger.info('L2: max=%f median=%f min=%f' % (np.max(scores), np.median(scores), np.min(scores)))
        return A, np.array(scores), np.mean(np.abs(Z), axis=0)

    def adjust_weights(self, W, A, J, mean_Z, weight_block_size=1):
        J = sorted(J)
        d = W.shape[1]
        w_norm = init_w_norm = torch.norm(W,dim=1).mean()
        W_ = W
        A_ = A
        for ji, j in enumerate(J):
            torch.cuda.ipc_collect()
            # print(ji, 'Aj = ', float((1-A[:,j]*A[j]).min()), float((1-A[:,j]*A[j]).max()))
            # print(ji, 'W =', float(W.min()), float(W.max()))
            # print('')

            j -= d-W.shape[1]
            print(ji, j, A.shape, W.shape)
            bJ = [i for i in range(W.shape[1]) if i != j]

            # method 1
            w_shape = W.shape
            if len(w_shape) == 4:
                permutation_order = [0,2,3,1]
                W = W.permute(permutation_order)
                permuted_w_shape = W.shape
                W = W.contiguous().view(-1,w_shape[1])
                reverse_permutation_order = [0,3,1,2]

            # w_update = (A[:,[j]].mm(W_.transpose(0,1)[[j]])).transpose(0,1)
            w_update = W[:,[j]].mm(A[:,[j]].transpose(0,1))
            update_norm = torch.norm(w_update, p=2, dim=1).mean()
            print('update_norm:', update_norm)         
            print('w_norm:',w_norm)
            future_error = (torch.abs(A[:, j]) * mean_Z[j]).mean()

            if (update_norm > 1 and update_norm > w_norm) or future_error > 1.0:# and ((ji) % weight_block_size == 0):
                print('update norm too large, breaking...')
                break

            W += w_update
            if len(w_shape) == 4:
                W = W.contiguous().view(permuted_w_shape)
                W = W.permute(reverse_permutation_order)

            W = W[:,bJ]            
            print('future error: %.4f' % future_error)
            # if (ji+1) % weight_block_size == weight_block_size-1 and future_error > 1.0:
            #     break

            # inv_eye = (1-torch.eye(A.shape[1], device=A.device))            
            # A_update = (A[:,[j]].mm(A[[j]]) * inv_eye)/(1-A[:,j]*A[j])
            A_update = A[:,[j]].mm(A[[j]])
            A_update[range(A.shape[0]), range(A.shape[0])] = 0            
            A_update /= (1-A[:,j]*A[j])

            A += A_update
            A = A[bJ][:,bJ]

            if weight_block_size == 1 or (ji > 0 and (ji+1) % weight_block_size == 0):
                print('caching W and A...')
                W_ = W
                A_ = A

            # if not torch.isfinite(A).any():
            #     print('A =',A)
            #     exit(0)

            # if not torch.isfinite(W).any():
            #     print(ji, 'Aj = ', (1-A[:,j]*A[j]).max(), (1-A[:,j]*A[j]).min())
            #     print(ji, 'W =',W)
            #     exit(0)
            
            w_norm = torch.norm(W,dim=1).mean()            
            # W, A = W_, A_
        print('change in weight norm: %.4f -> %.4f' % (init_w_norm, w_norm))
        print(W_.shape)
        return W_, A_.detach().cpu()
    
    def remove_input_neurons(self, layer_idx, neuron_idxs, A=None, mean_Z=None, weight_block_size=1):
        outsize, insize = utils.get_layer_input_output_size(self.layers[layer_idx])            
        retain_idxs = [i for i in range(insize) if i not in neuron_idxs]
        layer = self.layers[layer_idx]
        if len(retain_idxs) == 0:
            return layer
        W = layer.weight.data
        b = layer.bias.data
        if isinstance(layer, nn.Linear):
            if A is not None:
                if weight_block_size > 1:
                    h = w = int(np.sqrt(weight_block_size))
                    W = W.view(W.shape[0], -1, h, w)                    
                    b_ = b / (W.shape[2] * W.shape[3])
                    b_ = b_.view(-1,1,1,1)
                    b_ = b_.expand(-1,-1,W.shape[2], W.shape[3])
                    
                    W = torch.cat((W, b_), dim=1)                    
                    W, A = self.adjust_weights(W, A, neuron_idxs, mean_Z)
                    W, b = W[:,:-1], W[:,-1]
                    
                    b = b[:,0,0]
                    W = W.view(W.shape[0], -1)                    
                else:
                    W = torch.cat((W, b.view(-1,1)), dim=1)
                    W, A = self.adjust_weights(W, A, neuron_idxs, mean_Z, weight_block_size)
                    W, b = W[:,:-1], W[:,-1]
                    b = b.view(-1)
            else:
                W = W[:, retain_idxs]
            new_layer = nn.Linear(W.shape[1], outsize)
            new_layer.weight.data = W
            new_layer.bias.data = b
        elif isinstance(layer, nn.BatchNorm2d):
            W = W[retain_idxs]
            b = layer.bias[retain_idxs]
            running_mean = layer.running_mean[retain_idxs]
            running_var = layer.running_var[retain_idxs]

            new_layer = nn.BatchNorm2d(W.shape[0])
            new_layer.weight.data = W
            new_layer.bias.data = b
            new_layer.running_mean = running_mean
            new_layer.running_var = running_var
        elif isinstance(layer, nn.Conv2d):            
            if A is not None:
                b_ = b / (W.shape[2] * W.shape[3])
                b_ = b_.view(-1,1,1,1)
                b_ = b_.expand(-1,-1,W.shape[2], W.shape[3])

                W = torch.cat((W, b_), dim=1)
                W, A = self.adjust_weights(W, A, neuron_idxs, mean_Z, weight_block_size)
                W, b = W[:,:-1], W[:,-1]
                b = b[:,0,0]
            else:
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
        return new_layer, A

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

    def shrink_layer(self, i, data, factor=1, difference=0, pruned_neurons=None, A=None, mean_Z=None):        
        if i == len(self.layers)-1 or i == -1:
            raise IndexError('Can not shrink output layer')

        # if self.shrinking_state['layer_idx'] != i:
        #     self.update_shrinking_state(i, None, None)
        # A1 = self.shrinking_state['A1']
        # A2 = self.shrinking_state['A2']
        # print(A1)

        outsize, _ = utils.get_layer_input_output_size(self.layers[i])
        if pruned_neurons is None:
            new_size = int(outsize*factor - difference)
            if self.args.readjust_weights:            
                A, salience_scores, mean_Z = self.compute_prune_probability(i, data)
                # if self.is_last_conv(i):
                #     del A
                #     torch.cuda.ipc_collect()
                #     A, _, mean_Z = self.compute_prune_probability(i, data, flatten_conv_map=True)
                #     torch.cuda.ipc_collect()
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
        else:
            if self.args.readjust_weights and (A is None or mean_Z is None):            
                A, salience_scores, mean_Z = self.compute_prune_probability(i, data)
        
        k = i+1
        while k < len(self.layers) and not utils.is_weighted(self.layers[k]):
            k += 1
        # if utils.is_input_modifiable(self.layers[k]) and not isinstance(self.layers[k], nn.BatchNorm2d):
        _, old_insize = utils.get_layer_input_output_size(self.layers[k])
        nrepeats = old_insize // outsize
        
        # if self.is_last_conv(i):
        #     num_last_conv_filts = self.layers[i].weight.data.shape[0]            
        #     idxs = np.arange(old_insize).reshape(num_last_conv_filts, -1)
        #     pruned_input_neurons_ = pruned_input_neurons = idxs[pruned_neurons].flatten()
        # else:
        pruned_input_neurons_ = pruned_input_neurons = pruned_neurons
        
        if isinstance(self.layers[k], nn.BatchNorm2d):
            pruned_input_neurons = pruned_neurons

        init_w_norm = torch.norm(self.layers[k].weight.data, dim=-1).mean()
        if self.args.readjust_weights:
            if self.is_last_conv(i):
                # A2 = torch.repeat_interleave(A, repeats=nrepeats, dim=0)
                # A2 = torch.repeat_interleave(A2, repeats=nrepeats, dim=1)[:-nrepeats+1, :-nrepeats+1]
                # mean_Z2 = np.repeat(mean_Z, nrepeats) // nrepeats
                self.layers[k], _ = self.remove_input_neurons(k, pruned_input_neurons, A, mean_Z, nrepeats)
                # self.shrinking_state['A2'] = updated_A2
                torch.cuda.ipc_collect()
            else:
                self.layers[k], _ = self.remove_input_neurons(k, pruned_input_neurons, A, mean_Z)
                # self.shrinking_state['A1'] = updated_A1
                # print(updated_A1.shape)                      
        else:
            if self.is_last_conv(i):
                num_last_conv_filts = self.layers[i].weight.data.shape[0]            
                idxs = np.arange(old_insize).reshape(num_last_conv_filts, -1)
                pruned_input_neurons_ = pruned_input_neurons = idxs[pruned_neurons].flatten()
            self.layers[k], _ = self.remove_input_neurons(k, pruned_input_neurons)
        pruned_input_neurons = pruned_input_neurons_

        w_norm = torch.norm(self.layers[k].weight.data, dim=-1).mean()
        print('change in weight norm: %.4f -> %.4f' % (init_w_norm, w_norm))        

                # if not isinstance(self.layers[k], nn.BatchNorm2d):
                #     break
            # k += 1
        
        bn_idx = i+1
        while bn_idx < len(self.layers) and not isinstance(self.layers[bn_idx], nn.BatchNorm2d):
            bn_idx += 1        

        _, new_insize = utils.get_layer_input_output_size(self.layers[k])
        print(self.layers[k].weight.data.shape, old_insize, new_insize)
        
        if self.args.readjust_weights:
            if self.is_last_conv(i):
                num_removed_neurons = (old_insize-new_insize)//nrepeats 
                # self.A_cache.setdefault(i, [None,None])[0] = A[num_removed_neurons:, num_removed_neurons:]
            else:
                num_removed_neurons = old_insize-new_insize            
            self.layers[i] = self.remove_output_neurons(i, pruned_neurons[:num_removed_neurons])
            print('layer[i].shape:',self.layers[i].weight.shape)
            if bn_idx < len(self.layers) and isinstance(self.layers[bn_idx], nn.BatchNorm2d):
                self.layers[bn_idx], _ = self.remove_input_neurons(bn_idx, pruned_neurons[:num_removed_neurons])
        else:
            self.layers[i] = self.remove_output_neurons(i, pruned_neurons)
            
            if bn_idx < len(self.layers) and isinstance(self.layers[bn_idx], nn.BatchNorm2d):
                self.layers[bn_idx], _ = self.remove_input_neurons(bn_idx, pruned_neurons)

        new_outsize, insize = utils.get_layer_input_output_size(self.layers[i])
        return new_outsize < outsize

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

def alexnetCIFAR(num_classes, pretrained=False, feature_extraction=False, classifier_depth=1, **kwargs):
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
    