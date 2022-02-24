import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.nn.modules.module import _addindent
import h5py
import argparse
import glob
import os
import sys
import time
sys.path.append('models')

import model_x1_improved as model_x1
import model_x2


class deep_3d_inversion(object):
    def __init__(self, parsed):

        print('   _____ _____ _____ ____  _   _ ')
        print('  / ____|_   _/ ____/ __ \| \ | |')
        print(' | (___   | || |   | |  | |  \| |')
        print('  \___ \  | || |   | |  | | . ` |')
        print('  ____) |_| || |___| |__| | |\  |')
        print(' |_____/|_____\_____\____/|_| \_|')
                                                                                     
        self.cuda = torch.cuda.is_available()    

        device = parsed['device']
        self.superresolution = parsed['resolution']

        if ('cuda' in device):
            if (self.cuda == False):
                print('GPU not available. Computing in CPU')
                device = 'cpu'
            else:
                print(f"Computing on GPU {device}")

        if (device == 'cpu'):
            print("Computing on CPU")

        self.device = torch.device(device)
               
        self.ltau = np.array([0.0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0])

        self.variable = ["T", "v$_z$", "h", "log P", "$(B_x^2-B_y^2)^{1/2}$", "$(B_x B_y)^{1/2}$", "B$_z$"]
        self.variable_txt = ["T", "vz", "tau", "logP", "sqrtBx2By2", "sqrtBxBy", "Bz"]
        self.units = ["K", "km s$^{-1}$", "km", "cgs", "kG", "kG", "kG"]
        self.multiplier = [1.0, 1.e-5, 1.e-5, 1.0, 1.0e-3, 1.0e-3, 1.0e-3]

        self.z_tau1 = 1300.0


    def load_weights(self):

        if (self.superresolution == 1):
            self.checkpoint = 'models/weights_x1_improved.pth'
            self.normalization = 'models/normalization_x1_improved.npz'
            print("Generating output x1")
            print("Defining inversion NN...")
            self.model = model_x1.block(n_input_channels=112*4, n_output_channels=7*7).to(self.device)

        if (self.superresolution == 2):
            self.checkpoint = 'models/weights_x2.pth'
            self.normalization = 'models/normalization_x2.npz'
            print("Generating output x2")
            print("Defining inversion NN...")
            self.model = model_x2.block(n_input_channels=112*4, n_output_channels=7*7).to(self.device)

        
                                
        tmp = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
                                                
        self.model.load_state_dict(tmp['inv_state_dict'])        
        print("=> loaded checkpoint for inversion '{}'".format(self.checkpoint))     
        self.model.eval()
                                                                
        tmp = np.load(self.normalization)
        self.phys_min, self.phys_max = tmp['minimum'], tmp['maximum']

    def test_hinode(self, parsed):

        print(f"Reading input file {parsed['input']}")

        f = h5py.File(parsed['input'], 'r')

        self.stokes = f['stokes'][:,:,:,:]

        if (parsed['normalize'] is not None):
            x0, x1, y0, y1 = parsed['normalize']
            print(f"Data will be normalized to median value in box : {x0}-{x1},{y0}-{y1}")
            stokes_median = np.median(self.stokes[0,x0:x1,y0:y1,0:3])
        else:
            print(f"Data is already normalized")
            stokes_median = 1.0
        
        f.close()
    
        print(f"Transposing data")
        self.stokes = np.transpose(self.stokes, axes=(0,3,1,2))      

        _, n_lambda, nx, ny = self.stokes.shape

        nx_int = nx // 2**4
        ny_int = ny // 2**4
        nx = nx_int * 2**4
        ny = ny_int * 2**4
        
        print(f"Size of map {nx} x {ny}")
        print(f"Cropping map to range (0,{nx})-(0,{ny}) ")

        self.stokes = self.stokes[:,:,0:nx,0:ny]

        print(f"Normalizing data")
        
        self.stokes /= stokes_median

        self.stokes[1,:,:,:] /= 0.1
        self.stokes[2,:,:,:] /= 0.1
        self.stokes[3,:,:,:] /= 0.1

        self.stokes = np.expand_dims(self.stokes.reshape((4*n_lambda,nx,ny)), axis=0)
                                                       
        logtau = np.linspace(0.0, -3.0, 70)

        self.load_weights()
                
        print("Running neural network inversion...")

        start = time.time()

        # Do it in two steps in this case
        if (nx > 512):
            n = nx // 512
            idx = []
            for i in range(n+1):
                idx.append(i*512)
            if (nx % 512 != 0):
                idx.append(nx)
                extra = 1
            else:
                extra = 0

            print(f"Doing in {n} parts : {idx}")


            for i in range(n+extra):
                input = torch.as_tensor(self.stokes[0:1,:,idx[i]:idx[i+1],:].astype('float32')).to(self.device)
                print(f"{idx[i]} - {idx[i+1]}")
                if (i == 0):
                    with torch.no_grad():                
                        output_model = self.model(input)
                        
                    output_model = output_model.cpu().numpy()                        
                else:
                    with torch.no_grad():                
                        output_model1 = self.model(input)

                    output_model = np.concatenate([output_model, output_model1.cpu().numpy()], axis=-2)                    
        else:

            input = torch.as_tensor(self.stokes[0:1,:,:,:].astype('float32')).to(self.device)
                
            with torch.no_grad():                
                output_model = self.model(input)

            output_model = output_model.cpu().numpy()

        end = time.time()
        print(f"Elapsed time : {end-start} s - {1e6*(end-start)/(nx*ny)} us/pixel")

        # Transform the tensors to numpy arrays and undo the transformation needed for the training
        print("Saving results")
        output_model = np.squeeze(output_model)
        output_model = output_model * (self.phys_max[:,None,None] - self.phys_min[:,None,None]) + self.phys_min[:,None,None]    
        output_model = output_model.reshape((7,7,self.superresolution*nx,self.superresolution*ny))

        
        tmp = '.'.join(self.checkpoint.split('/')[-1].split('.')[0:2])
        f = h5py.File(f"{parsed['output']}", 'w')
        db_logtau = f.create_dataset('tau_axis', self.ltau.shape)
        db_T = f.create_dataset('T', output_model[0,:,:,:].shape)
        db_vz = f.create_dataset('vz', output_model[1,:,:,:].shape)
        db_tau = f.create_dataset('tau', output_model[2,:,:,:].shape)
        db_logP = f.create_dataset('logP', output_model[3,:,:,:].shape)
        db_Bx2_By2 = f.create_dataset('sqrt_Bx2_By2', output_model[4,:,:,:].shape)
        db_BxBy = f.create_dataset('sqrt_BxBy', output_model[5,:,:,:].shape)
        db_Bz = f.create_dataset('Bz', output_model[6,:,:,:].shape)
        db_Bx = f.create_dataset('Bx', output_model[4,:,:,:].shape)
        db_By = f.create_dataset('By', output_model[5,:,:,:].shape)

        Bx = np.zeros_like(db_Bz[:])
        By = np.zeros_like(db_Bz[:])
                    

        db_logtau[:] = self.ltau
        db_T[:] = output_model[0,:,:,:] * self.multiplier[0]
        db_vz[:] = output_model[1,:,:,:] * self.multiplier[1]
        db_tau[:] = output_model[2,:,:,:] * self.multiplier[2]
        db_logP[:] = output_model[3,:,:,:] * self.multiplier[3]
        db_Bx2_By2[:] = output_model[4,:,:,:] * self.multiplier[4]
        db_BxBy[:] = output_model[5,:,:,:] * self.multiplier[5]
        db_Bz[:] = output_model[6,:,:,:] * self.multiplier[6]

        A = np.sign(db_Bx2_By2[:]) * db_Bx2_By2[:]**2    # I saved sign(Bx^2-By^2) * np.sqrt(Bx^2-By^2)
        B = np.sign(db_BxBy[:]) * db_BxBy[:]**2    # I saved sign(Bx*By) * np.sqrt(Bx*By)

    # This quantity is obviously always >=0
        D = np.sqrt(A**2 + 4.0*B**2)
        
        ind_pos = np.where(B >0)
        ind_neg = np.where(B < 0)
        ind_zero = np.where(B == 0)
        Bx[ind_pos] = np.sign(db_BxBy[:][ind_pos]) * np.sqrt(A[ind_pos] + D[ind_pos]) / np.sqrt(2.0)
        By[ind_pos] = np.sqrt(2.0) * B[ind_pos] / np.sqrt(1e-1 + A[ind_pos] + D[ind_pos])
        Bx[ind_neg] = np.sign(db_BxBy[:][ind_neg]) * np.sqrt(A[ind_neg] + D[ind_neg]) / np.sqrt(2.0)
        By[ind_neg] = -np.sqrt(2.0) * B[ind_neg] / np.sqrt(1e-1 + A[ind_neg] + D[ind_neg])
        Bx[ind_zero] = 0.0
        By[ind_zero] = 0.0

        db_Bx[:] = Bx
        db_By[:] = By

        f.close()
        
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Fast 3D LTE inversion of Hinode datasets')
    parser.add_argument('-i', '--input', default=None, type=str,
                    metavar='INPUT', help='Input file', required=True)
    parser.add_argument('-o', '--output', default=None, type=str,
                    metavar='OUTPUT', help='Output file', required=True)
    parser.add_argument('-n', '--normalize', default=None, type=int, nargs='+',
                    metavar='OUTPUT', help='Output file', required=False)
    parser.add_argument('-d', '--device', default='cpu', type=str, 
                    metavar='DEVICE', help='Device : cpu/cuda:0/cuda:1,...', required=False)
    parser.add_argument('-r', '--resolution', default=1, type=int, choices=[1,2],
                    metavar='RESOLUTION', help='Resolution', required=False)

    parsed = vars(parser.parse_args())

    if (not os.path.exists(parsed['output'])):

        deep_network = deep_3d_inversion(parsed)

        deep_network.test_hinode(parsed)
    else:
        print(f"Output file {parsed['output']} already exists. Remove it to recompute.")
