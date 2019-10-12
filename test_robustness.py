import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
from models import *
import argparse
import logging
import utils
import sys
import os
from art.classifiers import PyTorchClassifier
from art.attacks import FastGradientMethod, SaliencyMapMethod
import matplotlib.pyplot as plt

class mDataset(Dataset):
    def __init__(self, inputs, targets):
        super(mDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets
        assert (len(inputs) == len(targets))
    def __getitem__(self, index):
        ax = self.inputs[index]
        if not isinstance(ax,torch.Tensor):
            if isinstance(ax,np.ndarray):
                ax = torch.from_numpy(ax).float()
            else:
                ax = torch.FloatTensor(ax)
        y = self.targets[index]
        return (ax, y)
    
    def __len__(self):
        return len(self.inputs)

def save_images(images, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i,img in enumerate(images):
        plt.imsave(os.path.join(outdir,'%d.png'%i), img)

def fgsm(classifier, inputs, true_targets, epsilon):        
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=inputs)    
    return x_test_adv

def saliency_map(classifier, inputs, true_targets, epsilon):        
    adv_crafter = SaliencyMapMethod(classifier, theta=epsilon, gamma=0.11)
    x_test_adv = adv_crafter.generate(x=inputs)    
    return x_test_adv

def resultsToString(results):
    s = ''
    for k,v in results.items():
        s += '%s : %.4f\n' % (k,v)
    return s

def main(args):
    attack_fn={
        'fgsm':fgsm,
        'saliency_map': saliency_map
    }
    _, val_dataset, input_shape, n_classes = utils.load_dataset(args.dataset)
    inputs = np.array([x.numpy() for x,_ in val_dataset])
    inputs /= 255    
    targets = np.array([int(y) for _,y in val_dataset])

    model = torch.load(args.model_file)
    model = model.eval().to(args.device)

    loss = torch.nn.CrossEntropyLoss()    
    classifier = PyTorchClassifier(model, loss, None, input_shape, n_classes, 
                                    preprocessing=(0,1/255), 
                                    clip_values=(0,1))

    base_preds = np.argmax(classifier.predict(inputs, 1024),1)
    results = {
        # 'baseline': utils.evaluate(model, val_dataset, args.device)
        'baseline': np.sum(base_preds == targets)/len(inputs)
    }        
    for attack in args.attacks:
        logger.info('Crafting Adversarial Examples Using %s' % attack)
        x_test_adv = attack_fn[attack](classifier, inputs, targets, args.epsilon)         
        # adv_dataset = mDataset(x_test_adv, targets)
        preds = np.argmax(classifier.predict(x_test_adv, 1024),1)
        acc = np.sum(preds == targets)/len(inputs)
        results[attack] = acc
        x_test_adv = np.transpose(x_test_adv, (0,2,3,1))#.astype('uint8')        
        save_images(x_test_adv, os.path.join(args.outdir, attack))

        # logger.info('adversarial_pred\tbaseline_pred\ttarget')
        # for ap,bp,t in zip(preds, base_preds, targets):
        #     logger.info('%d\t\t\t%d\t\t\t%d' % (ap,bp,t))

    logger.info(resultsToString(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--attacks', nargs='+', type=str)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--outdir', type=str, default='adv_examples/')
    parser.add_argument('--logfile', type=str, default='logs/attacks/attack.log')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    np.random.seed(1494)
    torch.manual_seed(1494)
    
    if not os.path.exists(os.path.dirname(args.logfile)):
        os.makedirs(os.path.dirname(args.logfile))
    if not os.path.exists(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(args.logfile),
    ])
    logger = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)    
    files_to_log = ["art.attacks.fast_gradient", "art.attacks.saliency_map"]
    for f in files_to_log:
        logger = logging.getLogger(f)
        logger.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info(args)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    main(args)