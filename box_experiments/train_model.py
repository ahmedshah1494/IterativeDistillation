import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
import numpy as np
from matplotlib.pyplot import imread,imsave
from itertools import product
from tqdm import trange,tqdm
import argparse
import sys
sys.path.append('../')
import distillation
from distillation import StudentModelWrapper
from copy import deepcopy
import logging

class SplitSingleOutput(nn.Module):
    def __init__(self):
        super(SplitSingleOutput, self).__init__()
    
    def forward(self, x):        
        x_ = 1-x
        new_x = torch.cat((x_+1e-10,x+1e-10), 1)
        return torch.log(new_x)

class Boost(nn.Module):
    def __init__(self, factor):
        super(Boost, self).__init__()
        self.factor = factor
    def forward(self, x):
        return x*self.factor

class Threshold(nn.Module):
    def __init__(self, v=0):
        super(Threshold, self).__init__()
        self.v=0
    def forward(self, x):
        x_ = (x >= self.v).float()
        return x_

class MLP(nn.Module):
    def __init__(self, nlayers, width, indim, outdim):
        super(MLP, self).__init__()
        layers = []
        for i in range(nlayers):
            if i == 0:
                indim_ = indim
            else:
                indim_ = width
            
            if i == nlayers-1:
                outdim_ = outdim
            else:
                outdim_ = width
            
            layers.append(nn.Linear(indim_, outdim_))
            if i != nlayers-1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
        layers.append(SplitSingleOutput())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

def train_batch(model, batch, optimizer, criterion, device):
    model = model.train()
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    z = model(x)
    loss = criterion(z,y)    
    loss.backward()
    optimizer.step()

    return z, float(loss.detach().cpu())

def evaluate(model, batch, criterion, device):
    model = model.eval()
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    z = model(x)
    metric = criterion(z,y)
    return z, metric

def KL(q,p):        
    loss1 = (p * torch.log((p+1e-8)/q)).view(-1)
    p_ = 1-p
    q_ = 1-q
    loss2 = (p_ * torch.log((p_+1e-8)/q_)).view(-1)
    loss = torch.mean(loss1+loss2)
    # print(p,q,loss1,loss2,loss)
    # if torch.isnan(loss):
    #     exit(0)
    return loss

def train(model, train_dataset, val_dataset, nepochs, batch_size, lr, device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)    

    model = model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    p = (list(model.parameters())[0])
    criterion = KL #lambda x,y: nn.KLDivLoss(reduction='batchmean')(torch.log(x),y)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    t = range(nepochs)    
    for i in range(nepochs):
        print('epoch '+str(i))
        avg_loss = 0
        t = tqdm(train_loader)

        for batch in t:
            model = model.train()
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z,y)            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss = float(loss)

            avg_loss = (avg_loss*i + loss)/(i+1) 
            t.set_postfix(loss=avg_loss)
        
        if i == nepochs-1:
            x,y = batch
            print(z[:10],y[:10], criterion(z[:10],y.to(device)[:10]))

        total_misses = 0
        total_samples = 0
        thresh = 0.9
        compute_misses = lambda x,y: torch.sum(torch.abs((x > thresh).long() - y.long()))
        for batch in val_loader:
            _,misses = evaluate(model, batch, criterion, device)
            total_misses += misses
            total_samples += 1 #batch[0].shape[0]
        
        error = float(total_misses)/total_samples
        print('val_error:',error)
        logger.info('> epoch=%d training_loss=%.4f validation_loss=%.4f' % (i, avg_loss, error))
def get_image_dataset(img_path,translate=False):
    img = imread(img_path)[:,:,0]  
    points = list(product(range(img.shape[0]), range(img.shape[1])))
    labels = [img[p[0],p[1]] for p in points]
    points = torch.from_numpy(np.array(points)).float()
    if translate:
        points -= torch.FloatTensor([[img.shape[0]//2, img.shape[1]//2]])
        points /= img.shape[0]//2
    labels = torch.from_numpy(np.array(labels).reshape(-1)).long()
    
    print(points.min(), points.max())
    print(torch.sum(labels))

    dataset = TensorDataset(points, labels)
    return dataset

def plot_img(img_name, model, dataset, device, translated=False):
    w = h = int(np.sqrt(len(dataset)))
    img = np.zeros((h, w, 3))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    sum_outputs = 0
    for batch in loader:
        z, _ = evaluate(model, batch, lambda x,y:None, device)
        x,y = batch
        
        # z = torch.argmax(z,1) 
        z = z[:,1]   
        z = torch.exp(z)
        # print(z)
        x = x.detach().cpu().numpy()
        z = z.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        sum_outputs += np.sum(z)

        if translated:
            x *= w//2
            x += np.array([[h//2,w//2]])

        x = x.astype(int)
        for i in range(x.shape[0]):
            img[x[i,0],x[i,1]] += z[i]
    img[img < 0] = 0
    img[img > 1] = 1 
    print(sum_outputs)
    imsave('%s.png' % img_name, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fat_mlp', action='store_true')
    parser.add_argument('--opt_mlp', action='store_true')
    parser.add_argument('--initialize_opt', action='store_true')
    parser.add_argument('--iterative_distillation', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--teacher_model', type=str)
    parser.add_argument('--img_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--nepochs', type=int, default=50)
    parser.add_argument('--train_layers', nargs='+', type=int, default=[0,2])
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--fat_mlp_width', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--shrink_factor', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--loss_weight', type=float, default=0.25)
    parser.add_argument('--tol', type=float, default=0.)
    parser.add_argument('--base_metric', type=float, default=0.1)
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--train_pc', type=float, default=0.8)
    args = parser.parse_args()

    np.random.seed(1494)
    torch.manual_seed(1494)
    torch.cuda.manual_seed_all(1494)

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(args.logfile),
        # logging.StreamHandler(sys.stdout)
    ])
    logger = logging.getLogger()

    logger.info(args)

    imgpath = 'box_in_box.png'
    dataset = get_image_dataset(imgpath,translate=True)
    train_len = int(args.train_pc * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, _ = random_split(dataset, [train_len, test_len])

    train_len = int(0.8*len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimal_mlp = nn.Sequential(
        nn.Linear(2,8),
        nn.ReLU(),
        nn.Linear(8,2),
        nn.ReLU(),
        nn.Linear(2,1),
        nn.Sigmoid(),
        SplitSingleOutput()
    )
    if args.model_path is not None:
        model = torch.load(args.model_path)
    elif args.opt_mlp:
        model = optimal_mlp
        for p in model.named_parameters():
            print(p)
        if args.initialize_opt:
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
            model[0].weight = nn.Parameter(torch.FloatTensor([
                [1,1],
                [-1,-1],
                [1,-1],
                [-1,1],
                [0,-1],
                [0,1],
                [-1,0],
                [1,0]
            ]))
            model[0].weight.requires_grad = False
            model[0].bias = nn.Parameter(torch.FloatTensor([
                1,
                1,
                1,
                1,
                0.5,
                0.5,
                0.5,
                0.5
            ]))
            model[3].weight = nn.Parameter(torch.FloatTensor([                                
                [0,0,0,0,1,1,1,1],
                [1,1,1,1,0,0,0,0]
            ]))
            model[3].weight.requires_grad = False
            model[3].bias = nn.Parameter(torch.FloatTensor([-4,-4]))
            model[6].weight = nn.Parameter(torch.FloatTensor([[-1,1]]))
            model[6].weight.requires_grad = False
            model[6].bias = nn.Parameter(torch.zeros(1))

            for p in model.named_parameters():
                print(p)
        # exit(0)
    elif args.fat_mlp:
        model = MLP(3,args.fat_mlp_width,2,1)
    model = model.to(args.device)
    print(model)
    if args.teacher_model is not None:
        teacher = torch.load(args.teacher_model, map_location='cpu')
        teacher = teacher.to(args.device)
        args_ = deepcopy(args)
        args_.C = 0
        args_.optimizer = 'adam'
        args_.batch_size = 128
        args_.soft_loss_type = 'xent'
        args_.outfile = args.model_name+'.pt'
        args_.train_on_student = False
        if args.iterative_distillation:
            base_metric = distillation.evaluate(teacher, val_dataset, args_)
            print('base_metric = %.4f' % (base_metric))
            train_logits = distillation.get_logits(teacher, train_dataset, args_)
            model = distillation.StudentModelWrapper(model)
            if args.fine_tune:
                    train(model, train_dataset, val_dataset, 20, args_.batch_size, args_.lr, args_.device)
            for i in args_.train_layers:                
                model = distillation.iterative_distillation(model, i, train_logits, val_dataset, val_dataset, 2, args_.base_metric, args_, logger, 
                # lambda x,y: -nn.functional.cross_entropy(x,y,reduction='sum')
                )
                if args.fine_tune:
                    train(model, train_dataset, val_dataset, 15, args_.batch_size, args_.lr, args_.device)
        else:
            distillation.simple_distillation(teacher, model, train_dataset, val_dataset, 2, args_)            
    else:
        train(model, train_dataset, val_dataset, args.nepochs, 128, args.lr, args.device)
    plot_img(args.img_name, model, dataset, args.device, translated=True)
    model = model.cpu()
    torch.save(model, "%s.pt" % args.model_name)


