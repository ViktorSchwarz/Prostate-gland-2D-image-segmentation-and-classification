### When executed, two images each containing 4 segmented frames are saved in a new directory.
import torch
import torchvision
import matplotlib.pyplot as plt
import os

from dataset import onezeromap
from Unet_segment import *

model_dir = 'trained_models'
# uploading models
model1 = UNet()
model2 = UNet()

model1.load_state_dict(torch.load(os.path.join(model_dir,'single_model1.pth'))['model_state_dict'])
model2.load_state_dict(torch.load(os.path.join(model_dir,'single_model2.pth'))['model_state_dict'])

# not exactly sure why this has to be done but
model1.eval()
model2.eval()

def single_pred(model, dataset, idx):
	''' returns the prediction, the consensus label and image itself
	with a path of that frame
	'''
	X,y = dataset[idx]
	X,y = torch.unsqueeze(X, dim=0), torch.unsqueeze(y, dim=0) # adding that one additional batch dimension

	pred = model(X)

	return (onezeromap(pred), y, X) , dataset.frame_paths[idx]

sample, frame = single_pred(model1, val_set, 200)

def visualise(sample, frame):
	''' saves an image of a sample with overlaid ground truth and prediction segmentations
	'''
	plt.figure()

	plt.imshow(sample[2][0,0,:,:], cmap='gray', vmin=0, vmax=255)
	# creting overlay masks
	mask_truth = np.ma.masked_where(sample[1][0,0,:,:] == 0, sample[1][0,0,:,:])
	mask_pred = np.ma.masked_where(sample[0][0,0,:,:] == 0, sample[0][0,0,:,:])

	plt.imshow(mask_truth, cmap='ocean', interpolation='none', alpha=0.7)
	plt.imshow(mask_pred, cmap='jet', interpolation='none', alpha=0.5)
	plt.title(frame[1:])
	plt.savefig(frame[1:]+'_overlay.png')

visualise(sample, frame)

def visualise_multiple(model, dataset, n_frames, save_name):
	''' saves n_frames of segmented frames with overlaid ground truth and predicted segmentation areas as one image
	n_frames is not used right now, instead the indicis array of 4 chosen frames is worked with 
	'''
	#indicis = np.random.randint(0, len(dataset), n_frames)
	indicis = [20,375,1000,745]
	plt.figure()

	for i, fr_idx in enumerate(indicis):
		sample, path = single_pred(model, dataset, fr_idx)

		plt.subplot(2,2, i+1)
		plt.imshow(sample[2][0,0,:,:], cmap='gray', vmin=0, vmax=255)
		mask_truth = np.ma.masked_where(sample[1][0,0,:,:] == 0, sample[1][0,0,:,:])
		mask_pred = np.ma.masked_where(sample[0][0,0,:,:] == 0, sample[0][0,0,:,:])
		#plt.imshow(sample[1][0,0,:,:], cmap='gray', vmin=0, vmax=1)
		plt.imshow(mask_truth, cmap='ocean', interpolation='none', alpha=0.7)
		plt.imshow(mask_pred, cmap='jet', interpolation='none', alpha=0.5)
		plt.ylabel(path[1:])

	# set the spacing between subplots
	plt.subplots_adjust(left=0.1,
	                    bottom=0.05, 
	                    right=0.9, 
	                    top=0.95, 
	                    wspace=0.1, 
	                    hspace=0.1)

	dir_path = 'segment_overlay'
	# saving to a new directory
	if not os.path.exists(dir_path):
		print("\\"+ dir_path,'directory created')
		os.makedirs(dir_path)

	plt.savefig(os.path.join(dir_path, save_name))


if __name__ == '__main__':
	visualise_multiple(model1, val_set, 4, 'model1_overlay.png')
	visualise_multiple(model2, val_set, 4, 'model2_overlay.png')

