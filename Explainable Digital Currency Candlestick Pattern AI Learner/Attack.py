import matplotlib.pyplot as plt
import mpl_finance as mpf
from tqdm import trange
import numpy as np

from utils import util_processor as util_pro
from utils import util_gasf as util_gaf


class Attacker:
    def __init__(self, model=None, inputs=None, atk_candle=None, atk_channel=None, saver_ls=None,
                 r=1.2, p=10.0, d=0, t=10, R=199, reset=5, batch=256, bounds=(-1, 1)):
        self.model = model
        self.gaf, self.label, self.ohlc = inputs
        self.n_samples = self.gaf.shape[0]
        self.atk_cdl = atk_candle
        self.atk_chl = atk_channel
        self.saver_ls = saver_ls
        self.r = r
        self.p = p
        self.d = d
        self.t = t
        self.R = R
        self.reset = reset
        self.batch = batch
        self.LB, self.UB = bounds
        assert 0 <= r <= 2

    def _batch_perturb(self, batch_data):
        # r = np.random.uniform(0.9, 1.1, self.batch)
        r = np.random.uniform(0.8, 1.2, self.batch)
        unp_v = batch_data[:, self.atk_cdl, self.atk_cdl, self.atk_chl]
        result = r * unp_v
        # legalize upper & lower bounds
        # result[result < self.LB] = unp_v[result < self.LB]
        # result[result > self.UB] = unp_v[result > self.UB]
        result[result < self.LB] = self.LB
        result[result > self.UB] = self.UB
        assert np.sum(self.LB <= result) == self.batch
        assert np.sum(self.UB >= result) == self.batch
        return result

    def _reconstruct_gaf(self, gaf):
        recon_gaf = np.zeros((10, 10, 4))
        phi = np.arccos(gaf.diagonal()) / 2
        ts_n = np.cos(phi)
        for c in range(4):
            # phi = np.arccos(gaf.diagonal()) / 2
            # ts_n = np.cos(phi)
            root_item = ((1 - ts_n[c]**2)**0.5).T
            recon_gaf[:, :, c] = np.outer(ts_n[c], ts_n[c]) - np.outer(root_item, root_item)
        return recon_gaf

    def attackAll(self):
        # suc_gaf_ls, org_ohlc_ls, suc_label_ls, suc_logit_ls, org_label_ls = [], [], [], [], []
        adv_res = dict()
        for saver_n in self.saver_ls:
            adv_res[saver_n] = []

        for i in trange(self.n_samples // self.batch):
            # select each batch data & label
            start = i * self.batch
            if self.n_samples % self.batch:
                batch_data, batch_ohlc = self.gaf[start:].copy(), self.ohlc[start:].copy()
                batch_label = self.label[start:].copy()
            end = (i+1) * self.batch
            batch_data, batch_ohlc = self.gaf[start: end].copy(), self.ohlc[start: end].copy()
            batch_label = self.label[start: end].copy()

            # create counter to record successful time
            suc_counter = np.zeros(self.batch)

            for _run in range(1, self.R+1):
                # reset
                if _run % self.reset == 0:
                    batch_data = self.gaf[start: end].copy()

                # perturb only diagonal values (batch)
                p_v = self._batch_perturb(batch_data)
                batch_data[:, self.atk_cdl, self.atk_cdl, self.atk_chl] = p_v

                # Reconstruct GAF
                batch_gaf = np.zeros_like(batch_data)
                for _b in range(self.batch):
                    batch_gaf[_b] = self._reconstruct_gaf(batch_data[_b])

                # pid = 0
                # plt.close()
                # fig = plt.figure()
                # ax1 = plt.subplot2grid((1, 2), (0, 0))
                # ax2 = plt.subplot2grid((1, 2), (0, 1))
                # # org ohlc > culr
                # batch_ohlc = self.ohlc[start: end].copy()
                # t_culr = np.expand_dims(batch_ohlc[pid], axis=0)
                # t_culr = util_gaf.ohlc2culr(t_culr)
                # # batch_gaf > culr
                # tmp_culr_ts = np.zeros((10, 4))
                # for c in range(4):
                #     tmp_culr_ts[:, c] = util_gaf.gasf2ts(batch_gaf[pid, :, :, c])
                # tmp_culr_ts = np.expand_dims(tmp_culr_ts, axis=0)
                # ts_arr2 = util_gaf.culr2ohlc(t_culr, tmp_culr_ts)

                # min_v = np.amin(np.array([batch_ohlc[pid], ts_arr2[0]]))
                # max_v = np.amax(np.array([batch_ohlc[pid], ts_arr2[0]]))

                # # plot arrays
                # ts_arr1 = np.c_[range(10), batch_ohlc[pid]]
                # ts_arr2 = np.c_[range(10), ts_arr2[0]]
                # mpf.candlestick_ohlc(ax1, ts_arr1, width=0.6, alpha=1,
                #                     colordown='#53c156', colorup='#ff1717')
                # mpf.candlestick_ohlc(ax2, ts_arr2, width=0.6, alpha=1,
                #                     colordown='#53c156', colorup='#ff1717')
                # ax1.set_ylim([min_v, max_v])
                # ax2.set_ylim([min_v, max_v])
                # org_val = t_culr[0, self.atk_cdl, self.atk_chl]
                # atk_val = 
                # ax1.set_title('original value: %s' % org_val, fontsize=12)
                # ax2.set_title('attacked value: %s' % atk_val, fontsize=12)
                # plt.show()
                # breakpoint()

                batch_pred = self.model.predict(batch_gaf)
                batch_class = batch_pred.argmax(axis=-1)

                cou_cond = (suc_counter == 0)
                chg_cond = (batch_class != batch_label)
                suc_cond = (cou_cond & chg_cond)
                suc_counter[suc_cond] = _run
                adv_res['suc_gaf'].extend([gaf for gaf in batch_gaf[suc_cond]])
                adv_res['org_ohlc'].extend([ohlc for ohlc in batch_ohlc[suc_cond]])
                adv_res['suc_label'].extend([l for l in batch_class[suc_cond]])
                adv_res['suc_logit'].extend([l for l in batch_pred[suc_cond]])
                adv_res['org_label'].extend([l for l in batch_label[suc_cond]])
                # print('*** %s-%s-run %s: %s' % (self.atk_cdl, self.atk_chl, _run, np.sum(suc_counter != 0)))

        return adv_res
