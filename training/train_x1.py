import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import h5py
import scipy.io as io
import scipy.ndimage as nd
from tqdm import tqdm
import sys
sys.path.append('../models')

import model_x1 as model
import time
import shutil
import nvidia_smi # pip install nvidia-ml-py3
import argparse
import os


class dataset_spot(torch.utils.data.Dataset):
    def __init__(self, n_training=10000, n_pixels=32, image_range=[0,10]):
        super(dataset_spot, self).__init__()

        noise_scale = 3e-3

        self.root = '/scratch1/aasensio/hinode'
        
        self.noise_scale = np.hstack([noise_scale*np.ones(112), 10*noise_scale*np.ones(112), 10*noise_scale*np.ones(112), 10*noise_scale*np.ones(112)])[:,None,None]

        #--------------------
        # REMPEL Stokes
        #--------------------
        print("Reading REMPEL Stokes parameters")
        f = h5py.File(f'{self.root}/rempel_vzminus_stokes_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape
        
        f.close()
        
        self.stokes = np.copy(stokes)        
        self.stokes[:,:,1,:] /= 0.1
        self.stokes[:,:,2,:] /= 0.1
        self.stokes[:,:,3,:] /= 0.1

        self.stokes = self.stokes.reshape((nx,ny,4*n_lambda))
        self.stokes = np.transpose(self.stokes, axes=(2,0,1))
       # self.stokes += np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes_inv.shape)
        

        #--------------------
        # REMPEL Stokes Inverted
        #--------------------
        print("Reading REMPEL Stokes parameters for inverted configuration")
        f = h5py.File(f'{self.root}/rempel_vzminus_stokes_invert_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape

        f.close()
        
        self.stokes_inv = np.copy(stokes)        
        self.stokes_inv[:,:,1,:] /= 0.1
        self.stokes_inv[:,:,2,:] /= 0.1
        self.stokes_inv[:,:,3,:] /= 0.1

        self.stokes_inv = self.stokes_inv.reshape((nx,ny,4*n_lambda))        
        self.stokes_inv= np.transpose(self.stokes_inv, axes=(2,0,1))
        #self.stokes_inv += np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes_inv.shape)

        #--------------------
        # CHEUNG Stokes
        #--------------------
        print("Reading CHEUNG Stokes parameters")
        f = h5py.File(f'{self.root}/cheung_vzminus_stokes_spat_spec_degraded_sir.h5', 'r')
        stokes = f['stokes'][:]
        nx, ny, _, n_lambda = stokes.shape

        f.close()
        
        self.stokes_cheung = np.copy(stokes)        
        self.stokes_cheung[:,:,1,:] /= 0.1
        self.stokes_cheung[:,:,2,:] /= 0.1
        self.stokes_cheung[:,:,3,:] /= 0.1
                
        self.stokes_cheung = self.stokes_cheung.reshape((nx,ny,4*n_lambda))
        self.stokes_cheung = np.transpose(self.stokes_cheung, axes=(2,0,1))
        #self.stokes_cheung += np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes_cheung.shape)

        #--------------------
        # REMPEL Model
        #--------------------        
        print("Reading REMPEL model parameters")
        f = h5py.File(f'{self.root}/rempel_vzminus_model_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,0:30])      # Substract the average height in the QS at tau=1

        f.close()
        
        self.phys = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys = np.max(self.phys, axis=(1,2))
        self.min_phys = np.min(self.phys, axis=(1,2))

        #--------------------
        # REMPEL Model Inverted
        #--------------------        
        print("Reading REMPEL model parameters in inverted configuration")
        f = h5py.File(f'{self.root}/rempel_vzminus_model_invert_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,0:30])      # Substract the average height in the QS at tau=1

        f.close()
        
        self.phys_inv = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys_inv = np.max(self.phys_inv, axis=(1,2))
        self.min_phys_inv = np.min(self.phys_inv, axis=(1,2))


        #--------------------
        # CHEUNG Model
        #--------------------        
        print("Reading CHEUNG model parameters")
        f = h5py.File(f'{self.root}/cheung_vzminus_model_spat_degraded_sir.h5', 'r')
        T = np.transpose(f['model'][:,:,1,:], axes=(2,0,1))
        vz = np.transpose(f['model'][:,:,3,:], axes=(2,0,1))
        tau = np.transpose(f['model'][:,:,0,:], axes=(2,0,1))
        logP = np.log10(np.transpose(f['model'][:,:,2,:], axes=(2,0,1)))
        Bx = np.transpose(f['model'][:,:,4,:], axes=(2,0,1))
        By = np.transpose(f['model'][:,:,5,:], axes=(2,0,1))
        Bz = np.transpose(f['model'][:,:,6,:], axes=(2,0,1))

        tau = tau - np.median(tau[0,0:30,20:50])      # Substract the average height in the QS at tau=1 (there is a patch in the corner that we avoid)

        f.close()
        
        self.phys_cheung = np.vstack([T, vz, tau, logP, np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2)), np.sign(Bx*By)*np.sqrt(np.abs(Bx*By)), Bz])
        self.max_phys_cheung = np.max(self.phys_cheung, axis=(1,2))
        self.min_phys_cheung = np.min(self.phys_cheung, axis=(1,2))

        #--------------------
        # Setup
        #--------------------        
        self.n_phys, self.nx_rempel, self.ny_rempel = self.phys.shape
        self.n_phys, self.nx_cheung, self.ny_cheung = self.phys_cheung.shape
        
        self.in_planes = 112 * 4
        self.out_planes = self.n_phys
        self.n_pixels = n_pixels
        self.n_training = n_training
        
        self.phys_max = np.max(np.vstack([self.max_phys,self.max_phys_inv,self.max_phys_cheung]), axis=0)
        self.phys_min = np.min(np.vstack([self.min_phys,self.min_phys_inv,self.min_phys_cheung]), axis=0)                

    def renoise(self):
        print("Reintroducing noise...")
        self.stokes_noise = self.stokes + np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes.shape)
        self.stokes_inv_noise = self.stokes_inv + np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes_inv.shape)
        self.stokes_cheung_noise = self.stokes_cheung + np.random.normal(loc=0, scale=self.noise_scale, size=self.stokes_cheung.shape)

        self.top = np.random.randint(0, self.nx_rempel - self.n_pixels, size=self.n_training)
        self.left = np.random.randint(0, self.ny_rempel - self.n_pixels, size=self.n_training)

        self.angle = np.random.randint(0, 4, size=self.n_training)
        self.flipx = np.random.randint(0, 2, size=self.n_training)
        self.flipy = np.random.randint(0, 2, size=self.n_training)

        self.flip_snapshot = np.random.randint(0, 3, size=self.n_training)

        # Since Cheung simulation has a different size, modify accordingly the ranges for the subpatches
        ind = np.where(self.flip_snapshot == 2)[0]
        n = len(ind)
        self.top[ind] = np.random.randint(0, self.nx_cheung - self.n_pixels, size=n)
        self.left[ind] = np.random.randint(0, self.ny_cheung - self.n_pixels, size=n)

        
    def __getitem__(self, index):

        if (self.flip_snapshot[index] == 0):
            input = self.stokes_noise[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            target = self.phys[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])            
        elif (self.flip_snapshot[index] == 1):
            input = self.stokes_inv_noise[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]

            target = self.phys_inv[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
                        
            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])
        else:
            input = self.stokes_cheung_noise[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
            
            target = self.phys_cheung[:,self.top[index]:self.top[index] + self.n_pixels, self.left[index]:self.left[index]+self.n_pixels]
                        
            target = (target - self.phys_min[:,None,None]) / (self.phys_max[:,None,None] - self.phys_min[:,None,None])


        input = np.rot90(input, self.angle[index], axes=(1,2)).copy()
        target = np.rot90(target, self.angle[index], axes=(1,2)).copy()
        
        if (self.flipx[index] == 1):
            input = np.flip(input, 1).copy()
            target = np.flip(target, 1).copy()            

        if (self.flipy[index] == 1):
            input = np.flip(input, 2).copy()
            target = np.flip(target, 2).copy()            

        # input += np.random.normal(loc=0, scale=self.noise_scale, size=input.shape)

        return input.astype('float32'), target.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')



class deep_3d_inversor(object):
    def __init__(self, batch_size, n_training=10000, n_validation=1000, n_pixels=32, gpu=0, synth_checkpoint='', vae_syn_checkpoint='', vae_mod_checkpoint=''):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
                
        if self.cuda:
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu) 
            print("Computing in {0} : {1}".format(nvidia_smi.nvmlDeviceGetName(self.handle), self.device))
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}

        print("Defining inversion NN...")
        self.model_inversion = model.block(n_input_channels=112*4, n_output_channels=7*7).to(self.device)

        print('   N. parameters inversion model : {0}'.format(sum(p.numel() for p in self.model_inversion.parameters() if p.requires_grad)))
        
        self.dataset_train = dataset_spot(n_training=n_training, n_pixels=n_pixels)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, 
                                                        shuffle=True, **kwargs)

        self.dataset_test = dataset_spot(n_training=n_validation, n_pixels=n_pixels)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size, 
                                                        shuffle=True, **kwargs)
        
    def optimize(self, epochs, lr=1e-4, smooth=0.05, gamma_model=1.0, gamma_stokes=0.0):

        self.lr = lr
        self.n_epochs = epochs
        self.smooth = smooth

        root = 'weights_x1_improved'

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = '{2}/{0}_-lr_{1}'.format(current_time, self.lr, root)

        print("Network name : {0}".format(self.out_name))

        # Copy model
        if not os.path.exists(root):
            os.makedirs(root)        
        shutil.copyfile(model.__file__, '{0}.model.py'.format(self.out_name))

        np.savez('{0}.normalization'.format(self.out_name), minimum=self.dataset_train.phys_min, maximum=self.dataset_train.phys_max)        

        self.optimizer = torch.optim.AdamW(self.model_inversion.parameters(), lr=self.lr,amsgrad=True)
        self.lossfn_L2 = nn.MSELoss().to(self.device)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                            # step_size=30,
                                            # gamma=0.5)

        self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=self.lr / 10.0)

        self.loss = []        
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name, self.lr), 'w')

        for epoch in range(1, epochs + 1):

            if (epoch == 1):
                self.dataset_train.renoise()
                self.dataset_test.renoise()
            
            self.train(epoch)
            self.test()

            self.scheduler.step()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = max(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'inv_state_dict': self.model_inversion.state_dict(),                
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr,                
            }, is_best, filename='{0}.pth'.format(self.out_name, self.lr))

        trainF.close()

    def train(self, epoch):
        self.model_inversion.train()
        print("Epoch {0}/{1} - {2}".format(epoch, self.n_epochs, time.strftime("%Y-%m-%d-%H:%M:%S")))
        t = tqdm(self.train_loader)
        
        loss_model_avg = 0.0
        loss_synth_avg = 0.0
        loss_avg = 0.0
        
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (data, target) in enumerate(t):

            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output_model = self.model_inversion(data)            
            
            loss = self.lossfn_L2(output_model, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model_inversion.parameters(), 0.1)

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                                                
            if self.cuda:
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle) 
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)

        self.loss.append(loss_avg)

    def test(self):
        self.model_inversion.eval()
                
        loss_model_avg = 0.0
        loss_synth_avg = 0.0
        loss_avg = 0.0
        
        n = 1
        t = tqdm(self.test_loader)

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(self.device), target.to(self.device)

                output_model = self.model_inversion(data)
                
                loss = self.lossfn_L2(output_model, target)
                
                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                if self.cuda:
                    tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle) 
                    t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
                else:
                    t.set_postfix(loss=loss_avg, lr=current_lr)

        self.loss_val.append(loss_avg)
            
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train neural network')    
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--epochs', '--epochs', default=50, type=int,
                    metavar='Epochs', help='Epochs')
    parsed = vars(parser.parse_args())

    deep_inversor = deep_3d_inversor(batch_size=64, n_training=50000, n_validation=2000, n_pixels=64, gpu=parsed['gpu'])
    deep_inversor.optimize(parsed['epochs'], lr=parsed['lr'], smooth=parsed['smooth'])