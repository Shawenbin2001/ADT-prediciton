from SAconvlstm import SAConvLSTM
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from utils import *
import math
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import pytorch_ssim as ssim
import time
import numpy as np
import os

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(855398)
       
        self.network = SAConvLSTM(configs.input_dim, configs.hidden_dim, configs.d_attn, configs.kernel_size).to(configs.device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
       
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.8, patience=0, verbose=True, min_lr=0.00001)

    def loss_sst(self, y_pred, y_true):
        
        rmse = torch.mean((y_pred - y_true)**2, dim=[3, 4])
        
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def train_once(self, wave, ratio):
       


        wave_pred = self.network(wave.float().to(configs.device), teacher_forcing=True, 
                                           scheduled_sampling_ratio=ratio, train=True)
 
        self.optimizer.zero_grad()
 
        loss_wave = self.loss_sst(wave_pred, wave[:, 1:].to(self.device))#
        loss_wave.backward()

        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
 
        self.optimizer.step()

        return loss_wave.item(), wave_pred

    def test(self, dataloader_test):
        wave_pred = []
        with torch.no_grad():
            for wave in dataloader_test: 

                wave= self.network(wave.float().to(configs.device)[:, :8], train=False)

                wave_pred.append(wave)
       
        return torch.cat(wave_pred, dim=0)

    def infer(self, dataset, dataloader):
        # calculate loss_func and score on a eval/test set
        self.network.eval()
        with torch.no_grad():
            wave_pred= self.test(dataloader)  
            wave_true = torch.from_numpy(dataset.data["dataset"][:, 8:]).float().to(self.device)

  
            loss_wave = self.loss_sst(wave_pred, wave_true).item()
 

            
            sc=[torch.sqrt(torch.mean((wave_pred[i,8:,0]-wave_true[i,8:,0])**2)).item() for i in range(len(wave_pred))]
            sc=-np.mean(np.array(sc))

        return loss_wave,sc
 

    def train(self, dataset_train, dataset_eval, chk_path):
        
        torch.manual_seed(855398)
        print('loading train dataloader')
        
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)
        
      
        count = 0
        best = - math.inf
        ssr_ratio = 1
        writer = SummaryWriter(self.configs.tensorboard_path)
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            epoch_losses=AverageMeter() 
            self.network.train()

            for j, (wave) in enumerate(dataloader_train): 
                
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                
                loss_wave, wave_pred = self.train_once(wave, ssr_ratio)###
               
                epoch_losses.update(loss_wave, len(wave))
               
                if j % self.configs.display_interval == 0:
                    sc = ssim.ssim(wave_pred[0,:], wave[0, 1:].float().to(configs.device)).item()
                    print('batch training loss: {:.2f}, score: {:.4f}, ssr ratio: {:.4f}'.format(loss_wave,sc, ssr_ratio))

            # evaluation
           
            writer.add_scalar("epoch_loss",epoch_losses.avg,i)
           
            loss_wave_eval,sc_eval= self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            writer.add_scalar("epoch_vail/loss",loss_wave_eval,i)
            writer.add_scalar("epoch_vail/sc",sc_eval,i)
            
            print('epoch eval loss:\nsst: {:.2f}, sc: {:.4f}'.format(loss_wave_eval, sc_eval))
           
            self.lr_scheduler.step(sc_eval)
            if sc_eval <= best:
                count += 1
               
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, sc_eval))
                self.save_model(chk_path)
                best = sc_eval

            if count == self.configs.patience:
    
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, path)




if __name__ == '__main__':
    start = time.time()
    
    print(configs.__dict__)

    print('\nreading data')
    
    dataset_train,dataset_eval=load_data()
    
    print('processing training set done')
    print('processing eval set done')
   
    trainer = Trainer(configs)
    trainer.save_configs('model.pkl')
    trainer.train(dataset_train, dataset_eval, 'model.chk')

    print('\n----- training finished -----\n')

    del dataset_train
    # del dataset_eval

    print('processing test set')
    print('loading test dataloader')
   
    dataloader_test = DataLoader(dataset_eval, batch_size=configs.batch_size_test, shuffle=False)
    
    chk = torch.load('model.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
   
    loss_wave_test,sc_test = trainer.infer(dataset=dataset_eval, dataloader=dataloader_test)
  
    print('test loss:\n wave: {:.2f}, score: {:.4f}'.format(loss_wave_test, sc_test))
    
    end = time.time()
    print("The function run time is : %.03f Min" %((end-start)/60))