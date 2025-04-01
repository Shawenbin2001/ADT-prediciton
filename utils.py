import numpy as np
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from pathlib import Path
import torch

def load_data():

	data = np.load(r"")
	data = np.swapaxes(data.reshape(-1, 20, 325, 378, 1), 2, 4)

	dict_train ={
	'dataset':data[0:3170]#[:1587],

	}

	dict_eval ={
	'dataset':data[3200:]#[1587:],
	
	}

	dataset_train=MyDataSet(dict_train)
	dataset_eval=MyDataSet(dict_eval)


	return dataset_train,dataset_eval

class MyDataSet(Dataset):

	def __init__(self, loaded_data):
		self.data = loaded_data


	def __len__(self):
		return len(self.data['dataset'])


	def __getitem__(self, idx):
		return self.data['dataset'][idx]


def load_testdata():
		data = np.load(r'..')
		data = np.swapaxes(data.reshape(-1, 20, 325, 378, 1), 2, 4)
		dict_test ={
		'dataset':data[:]}
		dataset_test=MyDataSet(dict_test)
		return dataset_test



def cutout(data,nb_patches):
	
	b,t,c,h,w=data.size()
	#
	x=data.view(b,t,c,nb_patches[0],int(h/nb_patches[0]),nb_patches[1],int(w/nb_patches[1])).permute(0,3,5,1,2,4,6).contiguous()
	
	x=x.view(-1,t,c,int(h/nb_patches[0]),int(w/nb_patches[1]))

	return x


def recut(data,batch_size,nb_patches):
    b,t,c,h,w=data.size()
    
    x=data.view(batch_size,nb_patches[0],nb_patches[1],t,c,h,w).permute(0,3,4,1,5,2,6).contiguous()

    
    x=x.view(1,t,c,int(h*nb_patches[0]),int(w*nb_patches[1]))
    return x 
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n 
		self.count += n
		self.avg = self.sum / self.count


