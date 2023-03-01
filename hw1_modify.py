import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

config = {
    'n_epochs': 8000,                
    'batch_size':275,
    'optimizer': 'Adam',           
    'optim_hparas': {                
        'lr': 5e-4,            
    },
    'lambda_1': 0.00075,
    'lambda_2': 0.00075,
    'feature':  [ 69, 51, 46, 47, 64, 36, 65, 54, 82, 83, 72, 35, 34, 52, 53, 70 ,71, 43 ]       ,

    'early_stop': 100,               
    'target_only' : True,
    'save_path': 'models/model.pth'
}

class MyModel(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear( input_dim , int(input_dim / 2  )  ),
            nn.LeakyReLU(),
            nn.Linear( int(input_dim / 2 ) ,  int(input_dim / 4 ) ),
            nn.LeakyReLU(),
            nn.Linear( int(input_dim / 4 ) ,  1 ),
        )
        
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        regularization_loss_ridged = 0
        regularization_loss_lasso = 0
        for param in self.parameters():
            regularization_loss_ridged += torch.sum( param ** 2 )
            regularization_loss_lasso += torch.sum( torch.abs(param) )
        
        return self.criterion(pred, target) + config['lambda_1'] * regularization_loss_ridged + \
                                                config['lambda_2'] * regularization_loss_lasso

import numpy as np
import csv
import os
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & valid loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['valid'])]
    figure(figsize=(16, 12))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['valid'], c='tab:cyan', label='valid')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig('Learning curve.png')

def plot_pred(dv_, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        error = 0
        for x, y in dv_:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    
        error = 0
        for i in range(len(preds)):
            error += (preds[i] - targets[i])**2
        import math
        print( 'MSE: ' ,math.sqrt(error/len(preds)))

    figure(figsize=(15, 15))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.savefig('Ground Truth v.s. Prediction.png')

class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=True):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(87))
        else:
            feats = config['feature']
        
        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            
            if mode == 'train':
                
                indices = [i for i in range(len(data))]
            elif mode == 'valid':
                indices = [i for i in range(len(data))]
            
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        
        self.data[:, 35:] = \
            (self.data[:, 35:] - self.data[:, 35:].mean(dim=0, keepdim=True)) \
            / self.data[:, 35:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
            .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        
        if self.mode in ['train', 'valid']:
            
            return self.data[index], self.target[index]
        else:
            
            return self.data[index]

    def __len__(self):
        
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            
    return dataloader

    
def train(tr_, dv_, model, config, device):
    ''' DNN training '''
    n_epochs = config['n_epochs']  
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas']) 

    min_mse = 1000.
    loss_record = {'train': [], 'valid': []}      
    early_stop_cnt = 0

    epoch = 0
    while epoch < n_epochs:
        model.train()                           
        for x, y in tr_:                     
            optimizer.zero_grad()               
            x, y = x.to(device), y.to(device)   
            pred = model(x)                     
            mse_loss = model.cal_loss(pred, y)  
            mse_loss.backward()                 
            optimizer.step()                    
            loss_record['train'].append(mse_loss.detach().cpu().item())

        valid_mse = valid(dv_, model, device)
        if valid_mse < min_mse:
            
            min_mse = valid_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['valid'].append(valid_mse)
        if early_stop_cnt > config['early_stop']:
            
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def valid(dv_, model, device):
    model.eval()                                
    total_loss = 0
    for x, y in dv_:                         
        x, y = x.to(device), y.to(device)       
        with torch.no_grad():                   
            pred = model(x)                     
            mse_loss = model.cal_loss(pred, y)  
        total_loss += mse_loss.detach().cpu().item() * len(x)  
    total_loss = total_loss / len(dv_.dataset)              

    return total_loss

def test(tt_, model, device):
    model.eval()                                
    preds = []
    for x in tt_:                            
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()     
    return preds

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))

    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
    number = 1
    while True:
        try:
            os.makedirs('./pred{}'.format(str(number)))
            break
        except:
            number+=1
            continue
    import shutil
    shutil.move('./pred.csv','./pred{}/pred.csv'.format(str(number)))
    shutil.move('./Learning curve.png','./pred{}/Learning curve.png'.format(str(number)))
    shutil.move('./Ground Truth v.s. Prediction.png','./pred{}/Ground Truth v.s. Prediction.png'.format(str(number)))
    shutil.copyfile('./HW1/model.py','./pred{}/model.py'.format(str(number)))
    shutil.copyfile('./HW1/__init__.py','./pred{}/config.py'.format(str(number)))



if __name__ == '__main__':
    myseed = 2022222
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    
    device = get_device()                 
    os.makedirs('models', exist_ok=True)  
    target_only = True                   

    tr_ = prep_dataloader('./covid_train.csv', 'train', config['batch_size'], target_only=config['target_only'])
    dv_ = prep_dataloader('./covid_train.csv', 'valid', config['batch_size'], target_only=config['target_only'])
    tt_ = prep_dataloader('./covid_test.csv', 'test', config['batch_size'], target_only=config['target_only'])
    
    model = MyModel(tr_.dataset.dim).to(device)  


    model_loss, model_loss_record = train(tr_, dv_, model, config, device)

    plot_learning_curve(model_loss_record, title='deep model')

    del model
    model = MyModel(tr_.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  
    model.load_state_dict(ckpt)
    plot_pred(dv_, model, device)  
    preds = test(tt_, model, device)  
    save_pred(preds, 'pred.csv')         