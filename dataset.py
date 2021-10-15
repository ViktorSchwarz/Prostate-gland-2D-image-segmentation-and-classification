# I will use this file as means of understanding the structure of the data
# and implementing the dataset and dataloader custom classes according to specification
# in the coursework section 4.
# Few words on how I have dealt with this problem, so before doing anything I split the dataset into training, test and validation subsets
# I have done so by first splitting 200 number case indicis that are inserted as an argument into a dataset class, this allows for the creation of 
# the different datasets by creating different instances of the dataset class with desired case indicis. Moreover, because we were asked for each case
# to sample the frames with equal probability, we first had to obtain the number of frames for each case and that is stored in the f_per_case tensor
# rest of the task was implemented in a pretty straight forward way as would be expected, I might have to go back to this dataset class and accomodate
# for the classification neural nets that will also have to be trained.

# The pixels of the images have values between 0 and 255 whereas the label map is either 0 or 1

import h5py 
import torch
import random
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

filename = 'dataset70-200.h5'
file = h5py.File(filename,'r')
im = file.get('label_0000_000_00')
#print(file['/frame_0000_000'])
#print(file.keys())
#print(torch.tensor(im)[40,:])
#print(torch.Tensor(im))

# indecis for splitting the dataset
train_idx, test_idx, val_idx = torch.utils.data.random_split(range(200), [120,40,40], generator=torch.Generator().manual_seed(42))

# data augmentation
transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomHorizontalFlip(p=1),
	transforms.ToTensor()
	])
# scaling the input images for a classifier
transform_resize = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((232,208)),
	transforms.ToTensor()
	])

def onezeromap(ten, data_load = False):
	'''converts the probability map into a map containing ones and zeros depending on
	whether the model thinks that pixel is a gland or not'''
	ones = torch.ones(ten.shape)
	zeros = torch.zeros(ten.shape)

	if use_cuda and not data_load: # if ten is on gpu ones and zeros also have to be
		ones = ones.cuda()
		zeros = zeros.cuda()

	return torch.where(ten>0.5, ones,zeros)
	
# dataset
class H5Dataset(torch.utils.data.Dataset):
	def __init__(self, filename, case_indicis, sampling = 'sampling1', transform = None, validate = False, binary = False):
		self.h5_file = h5py.File(filename, 'r')
		self.indicis = case_indicis
		self.sampling = sampling
		self.transform = transform
		self.binary = binary
		self.case_numbers = np.empty(int(len(self.h5_file.keys())/4), dtype = 'int64') # 5346 frames

		# loops over all the frames and stores the case number in that frames spot 
		for idx,name in enumerate(self.h5_file):
			if name.split('_')[0] == 'frame':
				self.case_numbers[idx] = int(name.split('_')[1])

		self.f_per_case = np.bincount(self.case_numbers) # array containig the number of frames per each case
		
		self.frame_paths = []     # paths to all the desired case frames
		self.validate = validate  # if this is true then frames are picked one by one  
		
		if validate:
			self.generate_frame_paths() 

	def __len__(self):
		# validation sets contain all the frames
		if self.validate:
			return len(self.frame_paths)
		else:
			return len(self.indicis)

	def __getitem__(self, idx):
		# binary classification
		if self.binary:
			# returns frames one by one,
			if self.validate:
				frame = torch.tensor(self.h5_file.get(self.frame_paths[idx])[()].astype('float32'))
				frame = transform_resize(frame)*255 # unsqueezes automaticaly, adds 1 chanel dimension

				# and a concensus label
				frame_idx = int(self.frame_paths[idx].split('_')[2]) 
				case_idx = int(self.frame_paths[idx].split('_')[1])

				label = self.label_consensus(case_idx, frame_idx) # 0 or 1

				return (torch.cat((frame, frame, frame), dim=0 ), torch.tensor([float(label)]) ) # returns the frame and label [1,58,52]

			# returns frames with frequency as specified in the task 
			else:
				case_idx = self.indicis[idx]
				frame_idx = random.randint(0, self.f_per_case[case_idx]-1) 

				frame = torch.tensor(self.h5_file.get('/frame_{0:04d}_{1:03d}'.format(case_idx,frame_idx))[()].astype('float32'))
				frame = transform_resize(frame)*255 # resizing the pictures, making them bigger
				label = self.label_consensus(case_idx, frame_idx ) # either 0 or 1

				if self.transform is not None:
					# had to multiply by the 255 to get it to the original greyscale
					# only apply the transforms 50 % of the time
					if random.uniform(0,1) > 0.5:
						frame = self.transform(frame)*255

				return (torch.cat((frame, frame, frame), dim=0 ), torch.tensor([float(label)])) # creating 3 channels, had convert it into floats, to match the dtype of network output
			
		# segmentation
		else:
			# returns frames one by one,
			if self.validate:
				frame = torch.tensor(self.h5_file.get(self.frame_paths[idx])[()].astype('float32'))

				# and a concensus label
				frame_idx = int(self.frame_paths[idx].split('_')[2]) 
				case_idx = int(self.frame_paths[idx].split('_')[1])

				label = self.label_consensus(case_idx, frame_idx)

				return (torch.unsqueeze(frame, dim=0), torch.unsqueeze(label, dim=0)) # returns the frame and label [1,58,52]

			# returns frames with frequency as specified in the task
			else:
				case_idx = self.indicis[idx]
				frame_idx = random.randint(0, self.f_per_case[case_idx]-1) 

				frame = torch.tensor(self.h5_file.get('/frame_{0:04d}_{1:03d}'.format(case_idx,frame_idx))[()].astype('float32'))

				if self.sampling == 'sampling1':
					label_idx = random.randint(0,2)
					label = torch.tensor(self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(case_idx,frame_idx,label_idx))[()].astype('float32'))

				elif self.sampling == 'sampling2':
					label = self.label_consensus(case_idx, frame_idx)

				if self.transform is not None:
					# had to multiply by the 255 to get it to the original greyscale
					# only apply the transforms 50 % of the time
					if random.uniform(0,1) > 0.5:
						#print('flipped')
						frame = torch.squeeze(self.transform(frame)*255)
						label = torch.squeeze(self.transform(label) )    

				return (torch.unsqueeze(frame, dim=0), torch.unsqueeze(label, dim=0))  # What this is does is really, adding a channel dimension, dataloader adds another dimension, which is the batch dimension

	def label_consensus(self, case_idx, frame_idx):
		'''returns the consensus label, returns 1 or 0 for classification and a map of zeros and ones for segmentation '''
		l1 = torch.tensor(self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(case_idx,frame_idx,0))[()].astype('int32'))
		l2 = torch.tensor(self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(case_idx,frame_idx,1))[()].astype('int32'))
		l3 = torch.tensor(self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(case_idx,frame_idx,2))[()].astype('int32'))
		l_avg = (l1+l2+l3)/3.0 # average

		# single binary classification 
		if self.binary:
			label_votes = [1 in l1, 1 in l2, 1 in l3]
			true_votes = label_votes.count(True)

			# contains protstate gland
			if true_votes >= 2:
				return 1
			# does not contain prostate gland
			else:
				return 0

		# segmentation
		else:
			return onezeromap(l_avg, data_load = True)

	def generate_frame_paths(self):
		''' Creates and array that will contain all the paths to frame indicis
		'''
		for case_idx in self.indicis:
			for frame_idx in range(self.f_per_case[case_idx]):
				self.frame_paths.append('/frame_{0:04d}_{1:03d}'.format(case_idx,frame_idx)) 


#dataset = H5Dataset(filename, val_idx, validate = True, binary = True)
#print(len(dataset))
#print(dataset.__getitem__(0)[0].shape)
#print(dataset[0][0][0,30,:])
#print(1 in torch.tensor([0,1,0,0]))
'''
img = dataset[0]

plt.figure()
plt.imshow(img[0][0,0,:,:], cmap='gray', vmin=0, vmax=255)
plt.savefig('142_img_aug.png')
plt.figure()
plt.imshow(img[1][0,0,:,:], cmap='gray', vmin=0, vmax=1)
plt.savefig('142_label_aug.png')
'''
