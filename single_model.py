# in this script we are training two models, one with sampling1 and second with sampling2 method
# after the training is finished we save the model parameters and arrays containing the train loss
# test loss and jaccard's metric to a new directory 'trained_models'
from Unet_segment import *
import os
import time 

use_cuda = torch.cuda.is_available() 

epochs = 5000  # 700 for result reproduction
model1 = UNet()
model2 = UNet()

# uses GPU if available
if use_cuda:
	model1.cuda()
	model2.cuda()

if __name__ == "__main__":	
	# training models
	time_start = time.time()

	print('Training model with sampling1 method')
	tr_loss1, tst_loss1, acc_jac1 = train(model1, epochs, train_loader1, test_loader1, loss_dice)
	print('Training model with sampling2 method')
	tr_loss2, tst_loss2, acc_jac2 = train(model2, epochs, train_loader2, test_loader2, loss_dice)

	models_dir = 'trained_models'

	# if the directory does not exist yet, creates a new one 
	if not os.path.exists(models_dir):
		print("\\"+ models_dir,'directory created')
		os.makedirs(models_dir)

	# saving models and their training progress
	torch.save({'model_state_dict': model1.state_dict(),
	            'train_loss': tr_loss1,
	            'test_loss': tst_loss1,
	            'jaccard_accuracy': acc_jac1
	           }, os.path.join(models_dir, 'single_model1.pth') )


	torch.save({'model_state_dict': model2.state_dict(),
	            'train_loss': tr_loss2,
	            'test_loss': tst_loss2,
	            'jaccard_accuracy': acc_jac2
	           }, os.path.join(models_dir, 'single_model2.pth') )


