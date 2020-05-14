from tqdm import tqdm, trange
import numpy as np
import pickle

from keras.models import load_model

# customized utilities
from utils import util_process as pro

# customized foolbox
from foolbox.criteria import TargetClassProbability
from foolbox.models import KerasModel
import foolbox


def generate_adversarial_examples(kmodel, fmodel, attacker, x_data, y_label):
	'''
    Args:
        pass

    Returns:
        pickle file: `dict` with the following keys
            'adversarial_x' (list):
            	each component is (numpy): (10, 10, 4)
            'adversarial_label' (list):
            	each component is (int)
	'''
	adversarial_data_to_save = {}
	adversarial_x = []
	adversarial_label = []

	for i in trange(len(x_data)):
		image = x_data[i]
		label = y_label[i]

		adversarial, fake_ls = attacker(image, label.argmax())
		try:
			adv_fpred = fmodel.forward_one(adversarial)
		except:
			adversarial_x.extend(fake_ls)
			adversarial_label.extend([label.argmax()] * len(fake_ls))
		
		if i % 1000 == 0:
			print("[ Info ] currently at data num :", i)

	adversarial_data_to_save['adversarial_x'] = adversarial_x
	adversarial_data_to_save['adversarial_label'] = adversarial_label

	# SAVE PICKLE
	with open('adversarial_fakedata.pkl', 'wb') as fp:
		pickle.dump(adversarial_data_to_save, fp)


def main():
	# load keras model
	kmodel = load_model('./models/cnn_model_10bar_ohlc.h5')
	# load data
	data = pro.load_pkl('./data/label8_eurusd_10bar_1500_500_val200_gaf.pkl')
	train_x = data['train_ohlc_gaf']
	train_label = data['train_label_onehot']
	# create foolbox model
	fmodel = KerasModel(kmodel, bounds=(-1, 1))
	# create our modified attack model
	MODIFIED_LocalSearchAttack = foolbox.attacks.LocalSearchAttack(model=fmodel)
	# generate fake data
	generate_adversarial_examples(kmodel = kmodel, fmodel = fmodel, attacker = MODIFIED_LocalSearchAttack,
	                              x_data = train_x, y_label = train_label)

if __name__ == '__main__':
    main()
