from keras.models import load_model
import matplotlib.pyplot as plt
import mpl_finance as mpf
from tqdm import tqdm
import numpy as np
import pickle

import foolbox
from foolbox.models import KerasModel
from foolbox.criteria import TargetClassProbability

# customized utilities
from utils import util_processor as pro


def test_adversarial(idx, kmodel, fmodel, attack, x_data, y_label, ohlc):
	gaf = x_data[idx, :, :, :]
	label = y_label[idx, :].argmax()
	ts = ohlc[idx, :, :]
	culr = pro.ohlc2culr(ohlc)[idx, :, :]
	culr = culr.reshape((1, *culr.shape))
	adv = attack(gaf, label)

	# If adv is None, then AssertionError
	assert adv is not None

	# If adv not None, check equeal or not
	if not np.array_equal(adv, gaf):
		# (1, 10, 4)
		culr_n = np.zeros((1, *ohlc.shape[1:]))
		for c in range(adv.shape[-1]):
			culr_n[:, :, c] = pro.gasf2ts(adv[:, :, c])
		culr = pro.culr2ohlc(culr_n, culr)[0, :, :]
		return 1
	else:
		return 0


def attack_all_samples(data, kmodel, fmodel, attack):
	results = []
	for l in range(1, 9):
		x_train, y_train, ts_train = pro.load_each_class(data, l)
		count = 0
		for i in tqdm(range(1500), desc='[ Info ] Label %s' % l):
			try:
				result = test_adversarial(idx=i, kmodel = kmodel, fmodel = fmodel, attack = attack,
							x_data = x_train, y_label = y_train, ohlc = ts_train)
				if result:
					count += 1
			except:
				pass
		results.append(count)
		print('\n    > Total results > Label %s : %s / 1500' % (l, count))
	# print all results
	for l, r in enumerate(results):
		print('\n    > Total results > Label %s : %s / 1500' % (l+1, r))
	return results


def main(params):
	# load data
	data = pro.load_pkl(params['pkl_name'])

	# load our model
	kmodel = load_model(params['model_name'])
	
	# create foolbox model
	fmodel = KerasModel(kmodel, bounds=(-1, 1))

	# customized LocalSearchAttack
	attack = foolbox.attacks.LocalSearchAttack(model=fmodel)

	# attack all samples
	results = attack_all_samples(data, kmodel, fmodel, attack)


if __name__ == '__main__':
	PARAMS = dict()
	PARAMS['pkl_name'] = './data/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl'
	PARAMS['model_name'] = './model/cnn_model_10bar.h5'
	
	main(PARAMS)
