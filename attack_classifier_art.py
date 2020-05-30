import torch
import numpy as np
import art
from art.utils import load_cifar10
from models import ModelWrapper, ModelWrapper2
from distillation import StudentModelWrapper, StudentModelWrapper2
import argparse
import utils
import os
import pickle

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model):
        super(NormalizationWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):        
        x = utils.normalize_image_tensor(x.float())
        return self.model(x)

def get_attack(classifier, args):
    attack_dict = {
        'pgdinf': lambda : art.attacks.ProjectedGradientDescent(classifier, norm=np.inf, 
                                        eps=args.eps, eps_step=args.eps_iter, 
                                        max_iter=args.max_iters, 
                                        num_random_init=args.nb_restarts, batch_size=128),
        'pgdl2': lambda : art.attacks.ProjectedGradientDescent(classifier, norm=2, 
                                        eps=args.eps, eps_step=args.eps_iter, 
                                        max_iter=args.max_iters, 
                                        num_random_init=args.nb_restarts, batch_size=128),
        'cwinf': lambda : art.attacks.CarliniLInfMethod(classifier, confidence=args.conf, learning_rate=args.lr, max_iter=args.max_iters, eps=args.eps, batch_size=128),
        'cwl2': lambda : art.attacks.CarliniL2Method(classifier, confidence=args.conf, learning_rate=args.lr, max_iter=args.max_iters, eps=args.eps, batch_size=128),
        'jsma': lambda : art.attacks.SaliencyMapMethod(classifier, theta=args.eps, gamma=args.gamma)
    }
    return attack_dict[args.attack]()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('model_path', type=str)
    parser.add_argument('--target_model_path', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datafolder', type=str, default='/home/mshah1/workhorse3/')
    parser.add_argument('--attack', type=str, default="none")
    parser.add_argument('--max_instances', type=int, default=-1)
    parser.add_argument('--nb_iter', type=int, default=20)
    parser.add_argument('--nb_restarts', type=int, default=1)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--eps_iter', type=float)
    parser.add_argument('--conf', type=float, default=50.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_iters', type=float, default=100)
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()

    if args.dataset.upper() == "CIFAR10" and args.attack in ["pgdinf", 'cwinf', 'jsma'] \
            and args.eps > 1.:
        args.eps = round(args.eps / 255., 4)


    if args.eps_iter is None:
        if args.attack in ["pgdinf", 'cwinf', 'jsma']:
            args.eps_iter = args.eps / 40.

    model = NormalizationWrapper(torch.load(args.model_path))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if args.dataset == 'cifar10':
        (_, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        # x_train = np.transpose(x_train, [0,3,1,2])
        x_test = np.transpose(x_test, [0,3,1,2])
        print(x_test.shape)
        classifier = art.classifiers.PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            channel_index=1,
            nb_classes=10,
        )

    
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # attack = art.attacks.FastGradientMethod(classifier=classifier, eps=0.2)    
    attack = get_attack(classifier, args)
    # print(vars(attack))
    x_test_adv = attack.generate(x=x_test)

    # Step 7: Evaluate the ART classifier on adversarial test examples

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

    outfile = os.path.join(os.path.dirname(args.model_path), 'advdata_%s_eps=%f.npy' % (args.attack, args.eps))
    with open(outfile, 'wb') as f:
        pickle.dump({
            'data': x_test_adv,
            'preds': predictions,
            'labels': y_test
        }, f)