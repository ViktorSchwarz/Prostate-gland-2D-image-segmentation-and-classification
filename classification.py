# File containing a densenet121 classifier adapted to detect a prostate gland,
# if the script is directly executed it trains the classifier and saves it 
# in the trained_models directory

import torch
import torchvision.models as models
from torch import nn
from dataset import *
from Unet_segment import *

densenet121 = models.densenet121(pretrained=True)

# freezing the training of the model parameters
for param in densenet121.parameters():
	param.requires_grad = False

# adapt the last layer
class_inputs = densenet121.classifier.in_features
class_outputs = 1

densenet121.classifier = nn.Sequential(
	nn.Linear(class_inputs, class_outputs),
	nn.Sigmoid()
	)

# dataset and dataloaders
train_set_cl = H5Dataset(filename, train_idx, transform = transform, binary = True)
train_loader_cl = DataLoader(train_set_cl,
	batch_size = 32,
	shuffle=True
	)

test_set_cl = H5Dataset(filename, test_idx, binary = True)
test_loader_cl = DataLoader(test_set_cl,
	batch_size = 1,
	shuffle=True
	)

val_set_cl = H5Dataset(filename, val_idx, validate = True, binary = True)
val_loader_cl = DataLoader(val_set_cl,
	batch_size = 1
	)

# cost function
bce_loss = nn.BCELoss()

def validate_composed(model_cl, model_seg, loader_cl, dataset_seg, loss_fn, treshold = 0.5):
	''' validates the segmentation model on the hold out test set (validation set) only on the samples
	that were classified as containing a gland
	'''
	jac_array = []
	losses = []

	with torch.no_grad():
		for idx, (X,y) in enumerate(loader_cl):
			# decide whether there is a gland in a first place
			y_pred_cl = model_cl(X)
			
			# it contains a gland -> segment
			if y_pred_cl > treshold:

				# obtaining the right label
				X, y = dataset_seg[idx]
				X, y = torch.unsqueeze(X, dim=0) , torch.unsqueeze(y, dim=0) # adding the batch dimension

				y_pred = model_seg(X)
				jac_array.append(jaccard(y_pred, y))

				loss = loss_fn(y_pred, y)
				losses.append(loss.item())
			else:
				continue

	return jac_array, losses


if __name__ == '__main__':
	### training procedure
	models_dir = 'trained_models'

	epochs = 1 # 400 for result reproduction
	train_loss, test_loss, test_acu = train(densenet121, epochs, train_loader_cl, test_loader_cl, bce_loss, classify = True)

	torch.save({'model_state_dict':densenet121.state_dict(),
		'train_loss': train_loss,
		'test_loss': test_loss,
		'test_accuracy': test_acu
		}, os.path.join(models_dir,'densenet121.pth') )
