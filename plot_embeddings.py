import torch
from torch.utils.data import DataLoader
from distillation import StudentModelWrapper, StudentModelWrapper2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import argparse
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
import os

def get_embeddings(model, loader,device):
    embeddings = []
    labels = []
    model = model.eval()
    model = model.to(device)
    for x,y in loader:
        x = x.to(device)
        z = model(x).detach().cpu().numpy()
        z = z.reshape(z.shape[0], -1)
        embeddings.append(z)
        labels.append(y.cpu().numpy())
    embeddings = np.concatenate(embeddings, 0)
    labels = np.concatenate(labels, 0)
    return embeddings, labels
        
        

def modify_model(model):
    layers = list(model.layers.children())[:-1]
    model.layers = torch.nn.Sequential(*layers)

def main(args):
    model = torch.load(args.model_file)
    if not isinstance(model, StudentModelWrapper2):
        model = StudentModelWrapper2(model, None, args)
    model = model.layers[:args.rep_layer+1]
    print(model)
    train_dataset, test_dataset, nclasses = utils.get_datasets(args)

    train_loader = DataLoader(train_dataset, 32, shuffle=False, num_workers=(cpu_count())//2)
    test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=(cpu_count())//2)

    # train_embeddings, train_labels = get_embeddings(model, train_loader, args.device)
    test_embeddings, test_labels = get_embeddings(model, test_loader, args.device)
    # embeddings = np.concatenate((train_embeddings,test_embeddings), 0)
    # labels = np.concatenate((train_labels,test_labels), 0)

    embeddings = test_embeddings
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_jobs=-1, verbose=2)
        embeddings = tsne.fit_transform(embeddings)
    labels = test_labels

    print(embeddings.shape)
    # sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":5})
    sns.set(font_scale=2, style='white')
    df = pd.DataFrame()
    df['x1'] = embeddings[:,0]
    df['x2'] = embeddings[:,1]
    df['Label'] = labels+1
    plt.figure(figsize=(20,10))
    sns.scatterplot(
        x="x1", y="x2",
        hue="Label",
        palette=sns.color_palette("hls", nclasses),
        data=df,
        legend="full",
        alpha=0.3
    )    
    outdir = 'embedding_plots/'
    outfile = os.path.basename(args.model_file).replace('.pt','_layer%d_embeddings.png'%args.rep_layer)
    plt.title(outfile.split('.')[0])
    plt.savefig(os.path.join(outdir, outfile))
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str,default="/home/mshah1/workhorse3")
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--model_file')
    parser.add_argument('--rep_layer', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)