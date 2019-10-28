import torch
from torch.utils.data import DataLoader
from distillation import StudentModelWrapper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import argparse
from multiprocessing import cpu_count

def get_embeddings(model, loader,device):
    embeddings = []
    labels = []
    model = model.eval()
    model = model.to(device)
    for x,y in loader:
        x = x.to(device)
        z = model(x).detach().cpu().numpy()
        embeddings.append(z)
        labels.append(y.cpu().numpy())
    embeddings = np.concatenate(embeddings, 0)
    labels = np.concatenate(labels, 0)
    return embeddings, labels
        
        

def modify_model(model):
    layers = list(model.layers.children())[:-1]
    model.layers = torch.nn.Sequential(*layers)

def main(args):
    model = torch.load('models/distilled/alexnet_cifar10_distilled.pt')
    modify_model(model)
    train_dataset, test_dataset, nclasses = utils.get_datasets(args)

    train_loader = DataLoader(train_dataset, 128, shuffle=False, num_workers=(cpu_count())//2)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=(cpu_count())//2)

    train_embeddings, train_labels = get_embeddings(model, train_loader, args.device)
    # test_embeddings, test_labels = get_embeddings(model, test_loader, args.device)
    # embeddings = np.concatenate((train_embeddings,test_embeddings), 0)
    # labels = np.concatenate((train_labels,test_labels), 0)

    embeddings = train_embeddings
    labels = train_labels

    print(embeddings.shape)
    # sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":5})
    sns.set(font_scale=2, style='white')
    df = pd.DataFrame()
    df['x1'] = embeddings[:,0]
    df['x2'] = embeddings[:,1]
    df['cluster'] = labels+1
    plt.figure(figsize=(20,10))
    sns.scatterplot(
        x="x1", y="x2",
        hue="cluster",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.savefig('cifar10-embeddings.png')
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)