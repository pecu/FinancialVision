from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import mse

import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np

# customized utilities
from utils import util_process as util_pro
from utils import util_plot as util_plt


class CVAE():
    def __init__(self, params):
        self.data = params['data']
        self.batch_size = params['batch_size']
        self.n_z = params['latent']
        self.n_bar = params['number_of_bars']
        self.n_input = self.n_bar * self.n_bar * 4

        # select optimizer
        optimizer = Adam(0.00001, 0.5)

        # build the encoder & decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        X = Input(shape=(self.n_input, ))
        cond = Input(shape=(1, ))
        self.mu, self.log_sigma = self.encoder([X, cond])
        z = Lambda(self.sample_z, output_shape=(self.n_z, ))([self.mu, self.log_sigma])
        z_cond = concatenate([z, cond])
        outputs = self.decoder(z_cond)

        self.cvae = Model([X, cond], outputs)
        self.cvae.compile(optimizer = optimizer, loss = self.vae_loss, metrics = [self.KL_loss, self.recon_loss])
        print(self.cvae.summary())

    def build_encoder(self):
        X = Input(shape=(self.n_input, ))
        cond = Input(shape=(1, ))

        # merge pixel representation and label
        inputs = concatenate([X, cond])
        h = Dense(1024)(inputs)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1024)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1024)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1024, activation='relu')(h)
        mu = Dense(self.n_z)(h)
        log_sigma = Dense(self.n_z)(h)

        return Model([X, cond], [mu, log_sigma])

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(self.batch_size, self.n_z), mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def build_decoder(self):
        d_in = Input(shape=(self.n_z + 1,))
        decoder_hidden_1 = Dense(1024)
        decoder_hidden_1_a = LeakyReLU(alpha=0.2)
        decoder_hidden_2 = Dense(1024)
        decoder_hidden_2_a = LeakyReLU(alpha=0.2)
        decoder_hidden_3 = Dense(1024)
        decoder_hidden_3_a = LeakyReLU(alpha=0.2)
        decoder_out = Dense(self.n_input, activation='tanh')

        h_p = decoder_hidden_1(d_in)
        h_p = decoder_hidden_1_a(h_p)
        h_p = decoder_hidden_2(h_p)
        h_p = decoder_hidden_2_a(h_p)
        h_p = decoder_hidden_3(h_p)
        h_p = decoder_hidden_3_a(h_p)
        outputs = decoder_out(h_p)

        return Model(d_in, outputs)

    def vae_loss(self, y_true, y_pred):
        recon = (mse(y_pred, y_true) * np.int(self.n_input))
        kl = K.sum(1 + self.log_sigma - K.square(self.mu) - K.exp(self.log_sigma), axis=-1) * -0.5
        return (recon + kl) * 0.5

    def KL_loss(self, y_true, y_pred):
        return K.sum(1 + self.log_sigma - K.square(self.mu) - K.exp(self.log_sigma), axis=-1) * -0.5

    def recon_loss(self, y_true, y_pred):
        return (mse(y_pred, y_true) * self.n_input)

    def train(self, epochs, batch_size, sample_interval):
        train_label = self.data['train_label'][:, 0]
        # get all label 1 - 8 ohlc data
        X_train = self.data['train_ohlc_gaf']
        y_train = train_label
        X_train = X_train.reshape((X_train.shape[0], self.n_input))

        kl_loss = []
        recon_loss = []
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            loss = self.cvae.train_on_batch([X_batch, y_batch], X_batch)       
            print("%d [loss: %f - KL_loss: %f - recon_loss: %f]" % (epoch, loss[0], loss[1], loss[2]))
            kl_loss.append(loss[1])
            recon_loss.append(loss[2])
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        # loss plot
        fig = plt.figure()
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        ax1.plot(kl_loss)
        ax2.plot(recon_loss)
        plt.show()

    def sample_images(self, epoch, is_gasd=False):
        # only sample label 1 & 2 for example here
        r, c = 3, 3
        v = np.zeros((r * c, self.n_z + 1))
        z = np.random.normal(size=(r * c, self.n_z))
        v[:, :100] = z

        # label 1
        v[:, -1] = 1
        gen_imgs_low = self.decoder.predict(v)
        gen_imgs_low = gen_imgs_low.reshape(gen_imgs_low.shape[0], self.n_bar, self.n_bar, 4)

        # label 2
        v[:, -1] = 2
        gen_imgs_high = self.decoder.predict(v)
        gen_imgs_high = gen_imgs_high.reshape(gen_imgs_high.shape[0], self.n_bar, self.n_bar, 4)

        self.plot_result(gen_imgs_low, './images/cvae/ep%s_label1.png' % epoch)
        self.plot_result(gen_imgs_high, './images/cvae/ep%s_label2.png' % epoch)

    def plot_result(self, gen_imgs, filename):
        # inverse to time-series
        temp = np.zeros((gen_imgs.shape[0], self.n_bar, 4))
        for i in range(gen_imgs.shape[0]):
            for c in range(gen_imgs.shape[-1]):
                gen_imgs_each = util_pro.gasf2ts(gen_imgs[i, :, :, c])
                temp[i, :, c] = gen_imgs_each
        gen_imgs = temp

        candle_open = gen_imgs[:, :, 0]
        candle_high = gen_imgs[:, :, 1]
        candle_low = gen_imgs[:, :, 2]
        candle_close = gen_imgs[:, :, 3]

        max_in_o_c = np.maximum(candle_open, candle_close)
        min_in_o_c = np.minimum(candle_open, candle_close)
        gen_imgs[:, :, 1] = np.maximum(candle_high, max_in_o_c)
        gen_imgs[:, :, 2] = np.minimum(candle_low, min_in_o_c)

        util_plt.plot_vae_result(gen_imgs, filename)

    def save_model(self):
        self.decoder.save('./models/cvae_decoder.h5')
        self.encoder.save('./models/cvae_encoder.h5')


if __name__ == '__main__':
    PARAMS = dict()
    PARAMS['number_of_bars'] = 10
    PARAMS['epochs'] = 10001  # 10000001
    PARAMS['latent'] = 100
    PARAMS['batch_size'] = 100
    PARAMS['sample_interval'] = 10000
    PARAMS['data'] = util_pro.load_pkl('./data/label8_eurusd_10bar_1500_500_val200_gaf.pkl')

    cvae = CVAE(PARAMS)
    cvae.train(epochs = PARAMS['epochs'], batch_size = PARAMS['batch_size'],
               sample_interval = PARAMS['sample_interval'])
    cvae.save_model()
