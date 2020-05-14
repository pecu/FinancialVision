from keras.models import load_model
import pandas as pd
import numpy as np

# customized utilities
from utils import util_process as util_pro
from utils import util_plot as util_plt


def save_csv(gen_imgs, idx):
    # inverse to time-series
    temp = np.zeros((gen_imgs.shape[0], 10, 4))
    for i in range(gen_imgs.shape[0]):
        for c in range(gen_imgs.shape[-1]):
            gen_imgs_each = util_pro.gasf2ts(gen_imgs[i, :, :, c])
            temp[i, :, c] = gen_imgs_each
    gen_imgs = temp

    candle_open = (gen_imgs[:, :, 0]).tolist()
    candle_high = (gen_imgs[:, :, 1]).tolist()
    candle_low = (gen_imgs[:, :, 2]).tolist()
    candle_close = (gen_imgs[:, :, 3]).tolist()
    
    df = pd.DataFrame(columns = ['open', 'high', 'low', 'close'])
    for i in range(len(candle_open)):
        zippedList =  list(zip(candle_open[i], candle_high[i], candle_low[i], candle_close[i]))
        df1 = pd.DataFrame(zippedList, columns = ['open', 'high', 'low', 'close']) 
        df = pd.concat([df, df1], axis=0)

    if idx == 1:
        global all_df
        all_df = df
    else:
        all_df = pd.concat([all_df, df], axis=0)

    if idx == 8:
        # all_df.to_excel('new_data/cvae_label_data.xlsx', index=False)
        all_df.to_excel('new_data/adversarial_fakedata.xlsx', index=False)


def sample_images(label, decoder):
        v = np.zeros((50, 101))
        z = np.random.normal(size=(50, 100))
        v[:, :100] = z
        v[:, -1] = label
        gen_imgs = decoder.predict(v)
        gen_imgs = gen_imgs.reshape(gen_imgs.shape[0], 10, 10, 4)
        save_csv(gen_imgs, label)


def main(params):
    decoder = load_model(params['cvae_decoder'])
    for label in range(1, 9):
        sample_images(label, decoder)


if __name__ == '__main__':
    PARAMS = dict()
    PARAMS['cvae_decoder'] = './models/cvae_decoder.h5'

    main(PARAMS)
