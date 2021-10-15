# Implemented UNet with residual blocks, plus a test and train loops with relevant metrics, loss functions and other functions
import torch

from torch import nn
from torch.utils.data import DataLoader
import time

from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()
print('Using {} device'.format(device))

######## UNet #########

class Resblock(nn.Module):
	def __init__(self, in_channels, n_feat):
		super(Resblock, self).__init__()
		self.in_channels = in_channels
		self.n_feat = n_feat

		self.block = torch.nn.Sequential(
			nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1, padding = 1, bias = False), # same convolution
			nn.BatchNorm2d(n_feat) ,
			nn.ReLU(),
			nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding = 1, bias = False), # same convolution
			nn.BatchNorm2d(n_feat)
			)
		
		self.relu = nn.ReLU()
		# identity transform residual connection
		if in_channels != n_feat:
			self.identity_transform = nn.Conv2d(in_channels, n_feat, kernel_size=1, stride=1)

	def forward(self, x):
		identity = x 
		
		x = self.block(x)

		# matching the dimensions (channels) of identity to output
		if self.in_channels != self.n_feat:
			identity = self.identity_transform(identity)

		x += identity
		return self.relu(x) 


class UNet(nn.Module):
	def __init__(self, in_ch=1, out_ch=1, features = [64, 128, 256]):
		super(UNet, self).__init__()
		self.features = features
		self.in_ch = in_ch
		self.out_ch = out_ch

		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.sigmoid = nn.Sigmoid()

		# Down
		for feature in self.features[:-1]:
			self.downs.append(Resblock(in_ch, feature))
			in_ch = feature

		# Up
		for feature in self.features[-2::-1]:
			self.ups.append(
				nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
			self.ups.append(
				Resblock(feature*2, feature))

		# bottleneck layer
		self.bottleneck = Resblock(self.features[-2], self.features[-1])

		# final output layer
		self.output = nn.Conv2d(self.features[0], self.out_ch, kernel_size=1)

	def forward(self, x):
		skip_conections = []

		# downsmpling
		for down in self.downs:
			x = down(x)
			skip_conections.append(x)
			x = self.pool(x)

		x = self.bottleneck(x)

		# upsampling 
		for i in range(0,len(self.ups),2):

			# transpose conv upsample
			x = self.ups[i](x)

			skip_con = skip_conections[-i//2-1]

			# matching the sample dimensons of upsampling to skip connection
			if x.shape != skip_con.shape:
				x = nn.functional.interpolate(x, size=skip_con.shape[2:]) # the new elements are copies of the nearest neighbour

			x = torch.cat((skip_con, x), dim=1)

			# convolutional resblock
			x = self.ups[i+1](x)

		x = self.output(x)

		return self.sigmoid(x)


# loss function
def loss_dice(y_pred, y_true, eps=1e-6):
	'''
	copied from the segmentation tutorial, soft dice coefficient loss function
	y_pred, y_true -> [N, C=1, H, W]
	'''
	numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2 
	denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps

	return torch.mean(1. - (numerator / denominator))

# metric for segmentation
def jaccard(y_pred, y_true, eps = 1e-6):
	'''
	Metric evaluating the performance of the unet, implemented for a single sample
	'''

	y_map = onezeromap(y_pred)
	intersection = torch.sum(y_map*y_true)

	y_add = y_map + y_true
	union = torch.sum(onezeromap(y_add))


	return (intersection)/(union+eps)

# metric for classification
def classify_accuracy(pred, y, treshold = 0.5):
	''' outputs 1 if classified correctly and 0 if incorrectly
	'''
	if y < treshold:
		if pred > treshold:
			return 0
		elif pred < treshold:
			return 1
	
	elif y > treshold:
		if pred > treshold:
			return 1
		elif pred < treshold:
			return 0 

def ceildiv(a, b):
	'''çeiling division'''
	return -(-a // b)

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    '''When called trains the model on one epoch of the data,  prints the total loss'''
    size = len(dataloader.dataset)
    # array that holds the loss of the given batch (last batch might have different number of samples)
    losses = torch.zeros(ceildiv(size, batch_size))


    for batch, (X, y) in enumerate(dataloader):
        # Pushing to GPU
        if use_cuda:
            X, y = X.cuda(), y.cuda()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # stores the total loss for the given batch, avgloss*#of samples
        losses[batch] = loss.item()*X.shape[0]

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_sum = torch.sum(losses)
    print(f"Train loss: {loss_sum:.3f}", end=" ")
    return loss_sum

def test_loop(dataloader, model, loss_fn, classify = False):
    '''Tests the model on the test set, prints relevant metrics'''
    size = len(dataloader.dataset)
    test_loss = 0
    acu = torch.zeros(size)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            if use_cuda:
                X, y = X.cuda(), y.cuda()

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # classification
            if classify:
                acu[batch] = classify_accuracy(pred, y)
            # segmentation
            else:
                acu[batch] = jaccard(pred, y)

    test_loss /= size
    acu_avg = torch.sum(acu)/size

    print(f"-- Test Accuracy: {(100*acu_avg):0.2f}%, Avg loss: {test_loss:.3f}", end=" ")
    return  test_loss, acu_avg*100

# creating datasets
train_set1 = H5Dataset(filename, train_idx, 'sampling1', transform)
train_loader1 = DataLoader(train_set1,
	batch_size = 32,
	shuffle=True
	# specifying num of workers to 2 produced an error, so I just got rid of it, migth want to look into this later
	)

test_set1 = H5Dataset(filename, test_idx, 'sampling1')
test_loader1 = DataLoader(test_set1,
	batch_size = 1,
	)

train_set2 = H5Dataset(filename, train_idx, 'sampling2', transform)
train_loader2 = DataLoader(train_set2,
	batch_size = 32,
	shuffle=True
	)

test_set2 = H5Dataset(filename, test_idx, 'sampling2')
test_loader2 = DataLoader(test_set2,
	batch_size = 1,
	)

val_set = H5Dataset(filename, val_idx, validate = True)
val_loader = DataLoader(val_set,
	batch_size = 1)

##### Training procedure ######
def train(model, epochs, train_loader, test_loader, loss_fn, classify = False):
	''' function that trains the model according to specified parameters
	works for both, classifier nad segmentor, returns the training history of the model which
	are arrays containing the metrics and losses of each epoch'''
	train_losses = torch.zeros(epochs)
	test_metric = torch.zeros(epochs)
	test_losses = torch.zeros(epochs)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	for epoch in range(epochs):
		print(f"Epoch {epoch+1} -------------------------------")
		start_time = time.time()
		train_losses[epoch] = train_loop(train_loader, model, loss_fn, optimizer, train_loader.batch_size)
		test_losses[epoch], test_metric[epoch] = test_loop(test_loader, model, loss_fn, classify = classify)
		print('-- time:{:.1f}s'.format(time.time()-start_time))
	print("Done!")

	return (train_losses, test_losses, test_metric)

# validation functions
def validate(model, val_loader, loss_fn, classify = False):
	''' loops over the validation set and returns the performance of the model, i.e. metric and losses'''
	metric_array = torch.zeros(len(val_loader.dataset))
	losses = torch.zeros(len(val_loader.dataset))

	for idx, (X,y) in enumerate(val_loader):
		y_pred = model(X)

		# classification
		if classify:
			metric_array[idx] = classify_accuracy(y_pred, y)
		# segmentation
		else:
			metric_array[idx] = jaccard(y_pred, y)

		loss = loss_fn(y_pred, y)
		losses[idx] = loss.item()

	return metric_array, losses

def bland_altman(model, val_loader):
	''' function that iterates through all the validation set samples and for each sample
	it takes a difference between the ground truth and prediction and divides it by an average
	which is: (pred-truth)/((pred+truth)/2)
	'''
	differences = torch.zeros(len(val_loader.dataset)) 
	averages = torch.zeros(len(val_loader.dataset))

	for idx, (X,y) in enumerate(val_loader):
		y_pred = model(X)
		y_map = onezeromap(y_pred)

		dif = torch.sum(y_map) - torch.sum(y)
		average = torch.sum(y_map+y)/2

		averages[idx] = average
		differences[idx] = dif/average 

	return differences*100, averages # convert to percentages

if __name__ == "__main__":
	#epochs = 1
	#model = UNet()
	#train(model, epochs, train_loader1, test_loader1, loss_dice)
	pass

