# This piece of code when executed trains the 5 different Unets for both sampling1 and sampling 2 methods, and saves the models
# training on CPU, because of a lack of access to a GPU I did not experiment with ensemble learning and merely wrote a code
# with a simplest ensemble learning implementation 

from Unet_segment import *
import os

# lists that will store the models
models_spl1 = nn.ModuleList()
models_spl2 = nn.ModuleList()

# creating models, they only differ by number of downsampling levels 2 and 3 
for i in range(3):
	models_spl1.append(UNet())
	models_spl2.append(UNet())

# adding an extra downsampling level with 512 features
for i in range(2):
	models_spl1.append(UNet(features = [64, 128, 256, 512]))
	models_spl2.append(UNet(features = [64, 128, 256, 512]))


if __name__ == "__main__":
	# training procedure
	epochs = 1 
	for model in models_spl1:
		train(model, epochs, train_loader1, test_loader1, loss_dice)

	for model in models_spl2:
		train(model, epochs, train_loader2, test_loader2, loss_dice)

	# saving the models
	weights1 = dict()
	weights2 = dict()

	for i in range(len(models_spl1)):
		weights1['model_state_dict{}'.format(i+1)] = models_spl1[i].state_dict()
		weights2['model_state_dict{}'.format(i+1)] = models_spl2[i].state_dict()

	models_dir = 'trained_models'
	# if the directory does not exist yet, creates a new one 
	if not os.path.exists(models_dir):
		print("\\"+ models_dir,'directory created')
		os.makedirs(models_dir)

	torch.save(weights1, os.path.join(models_dir, 'ensemble_model1.pth') )
	torch.save(weights2, os.path.join(models_dir, 'ensemble_model2.pth') )



