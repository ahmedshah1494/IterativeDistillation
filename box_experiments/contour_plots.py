import numpy as np
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from train_model import Boost, SplitSingleOutput, train, get_image_dataset, evaluate, plot_img, Threshold, MLP
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams.update({'font.size': 22})

def get2DProjection(insize, outsize):
    p = np.random.rand(outsize,insize)
    u,s,v = np.linalg.svd(p, full_matrices=False)
    print(u.shape, s.shape, v.shape)
    s /= s
    p_ = np.dot(u * s, v)
    p_inv = np.linalg.pinv(p_)
    return p_, p_inv

def initialize_handcrafted_net(W, threshold):
    W1 = W[:16].reshape(8,2)
    b1 = W[16:24]
    W2 = W[24:40].reshape(2,8)
    b2 = W[40:42]
    W3 = W[42:44].reshape(1,2)
    b3 = W[44:45]

    model = nn.Sequential(
                nn.Linear(2,8),
                Boost(50),
                nn.Sigmoid(),
                nn.Linear(8,2),
                Boost(50),
                nn.Sigmoid(),
                nn.Linear(2,1),
                Boost(10),
                nn.Sigmoid(),
                SplitSingleOutput()
            )

    if threshold:
        model = nn.Sequential(
                nn.Linear(2,8),
                Boost(1),
                Threshold(),
                nn.Linear(8,2),
                Boost(1),
                Threshold(),
                nn.Linear(2,1),                
                SplitSingleOutput()
            )

    model[0].weight = nn.Parameter(torch.FloatTensor(W1))
    model[0].weight.requires_grad = False
    model[0].bias = nn.Parameter(torch.FloatTensor(b1))
    model[3].weight = nn.Parameter(torch.FloatTensor(W2))
    model[3].weight.requires_grad = False
    model[3].bias = nn.Parameter(torch.FloatTensor(b2))
    model[6].weight = nn.Parameter(torch.FloatTensor(W3))
    model[6].weight.requires_grad = False
    model[6].bias = nn.Parameter(torch.FloatTensor(b3))

    return model

def initialize_custom_net(W, width):
    W1 = W[:2*width].reshape(width, 2)
    b1 = W[2*width:3*width]
    W2 = W[3*width:3*width+width**2].reshape(width,width)
    b2 = W[3*width+width**2:4*width+width**2]
    W3 = W[4*width+width**2:5*width+width**2].reshape(1,width)
    b3 = W[5*width+width**2:5*width+width**2+1]

    model = MLP(3, width, 2, 1)
    model.mlp[0].weight = nn.Parameter(torch.FloatTensor(W1))
    model.mlp[0].weight.requires_grad = False
    model.mlp[0].bias = nn.Parameter(torch.FloatTensor(b1))
    model.mlp[2].weight = nn.Parameter(torch.FloatTensor(W2))
    model.mlp[2].weight.requires_grad = False
    model.mlp[2].bias = nn.Parameter(torch.FloatTensor(b2))
    model.mlp[4].weight = nn.Parameter(torch.FloatTensor(W3))
    model.mlp[4].weight.requires_grad = False
    model.mlp[4].bias = nn.Parameter(torch.FloatTensor(b3))

    return model

def initialize_net(W, width=None, threshold=False):
    if width is None:
        return initialize_handcrafted_net(W,threshold)
    return initialize_custom_net(W,width)

def get_loss(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    total_misses = 0
    total_samples = 0
    for batch in loader:
        _,misses = evaluate(model, batch, criterion, device)
        total_misses += misses
        total_samples += 1

    error = float(total_misses)/total_samples
    return error

def main():
    np.random.seed(9999)
    torch.manual_seed(9999)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgpath = 'box_in_box.png'
    dataset = get_image_dataset(imgpath, translate=True)    
    train_len = int(0.02 * len(dataset))
    test_len = len(dataset) - train_len
    sample_dataset, _ = random_split(dataset, [train_len, test_len])

    if width == 1024:
        model = torch.load('2x1024_2pc.pt')
    else:
        model = torch.load('opt_initialized.pt')    
    params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    params = np.concatenate(params, 0)      
    model = initialize_net(params, width, threshold)

    grid_size = 100
    grid_range = np.linspace(-grid_size, grid_size+1, 25)    
    grid = list(product(grid_range, grid_range))

    model = model.to(device)
    print(get_loss(model, sample_dataset, device))
    for i in range(3):
        p, p_inv = get2DProjection(len(params), 2)
        p_inv, _ = get2DProjection(2, len(params))
        points = []
        values = []   
        opt_point = p.dot(params) 
        if normed:
            opt_point /= np.linalg.norm(opt_point)
        for x,y in grid:
            delta = p_inv.dot(np.array([x,y]))
            new_params = params + delta
            points.append(p.dot(new_params))

            model = initialize_net(new_params, width, threshold)
            model = model.to(device)
            loss = get_loss(model, sample_dataset, device)
            values.append(loss)

        points = np.array(points)
        values = np.array(values)
        X = points[:,0].reshape(int(np.sqrt(points.shape[0])),-1)
        Y = points[:,1].reshape(int(np.sqrt(points.shape[0])),-1)
        Z = values.reshape(int(np.sqrt(values.shape[0])),-1)

        cp = plt.contourf(X, Y, Z, cmap='gnuplot', levels=256)
        # plt.clabel(cp, inline=True, 
        #   fontsize=10)
        plt.colorbar(cp)
        plt.plot(opt_point[0], opt_point[1], 'r+', markersize=20)        
        if normed:
            plt.savefig('contor-%s%s%d-normed.png' % (str(width) if width is not None else '', 'thresh' if threshold else '', i))
        else:
            plt.savefig('contor%s%s%d-test.png' % (str(width) if width is not None else '', 'thresh' if threshold else '', i))
        plt.clf()
    model = initialize_net(params,width, threshold)
    model = model.to(device)
    plot_img('test', model, dataset, device, translated=True)

if __name__ == '__main__':
    normed = False
    width = 1024
    threshold = False
    main()

