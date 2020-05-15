import matplotlib.pyplot as plt
import mpl_finance as mpf
import numpy as np


def plot_error_result(y_pred=None, ts_data=None, samples=None, true_label=None, pred_label=None):
    '''
    Args:
        y_pred (numpy): (error_n, )
        ts_data (numpy): (error_n, 32, 4)
        samples (int): how many data to be plot
        label (int): which label to focus
    '''
    counter = 0
    for i in range(y_pred.shape[0]):
        # break if enough samples
        if counter >= samples:
            break
        counter += 1

        fig = plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        ts_arr = np.c_[range(ts_data[i, :, :].shape[0]), ts_data[i, :, :]]
        mpf.candlestick_ohlc(ax, ts_arr, width=0.6, alpha=1,
                             colordown='#53c156', colorup='#ff1717')
        title_str = 'True label: %s | Pred label: %s' % (true_label, pred_label)
        ax.set_title(title_str, fontsize=12)
        plt.show()


def plot_result(ts_data, epoch):
    '''
    Args:
        ts_data (numpy): (error_n, 32, 4)
    '''
    fig = plt.figure(figsize=(25, 25))
    axes_ls = []

    for l in range(3):
        for j in range(3):
            axes_ls.append(plt.subplot2grid((3, 3), (l, j), colspan=1, rowspan=1))
    
    for i in range(ts_data.shape[0]):
        ts_arr = np.c_[range(ts_data[i, :, :].shape[0]), ts_data[i, :, :]]
        mpf.candlestick_ohlc(axes_ls[i], ts_arr, width=0.4, alpha=1,
                            colordown='#53c156', colorup='#ff1717')
        axes_ls[i].axes.get_yaxis().set_visible(True)
        axes_ls[i].axes.get_xaxis().set_visible(True)
        axes_ls[i].axes.get_yaxis().set_ticks([])
        axes_ls[i].axes.get_xaxis().set_ticks([])

    plt.close()
    fig.savefig("./images/vocal_%d.png" % epoch)
    plt.close()


def plot_vae_result(ts_data, filename):
    '''
    Args:
        ts_data (numpy): (error_n, 32, 4)
    '''
    fig = plt.figure(figsize=(25, 25))
    axes_ls = []

    for l in range(3):
        for j in range(3):
            axes_ls.append(plt.subplot2grid((3, 3), (l, j), colspan=1, rowspan=1))
    
    for i in range(ts_data.shape[0]):
        ts_arr = np.c_[range(ts_data[i, :, :].shape[0]), ts_data[i, :, :]]
        mpf.candlestick_ohlc(axes_ls[i], ts_arr, width=0.4, alpha=1,
                            colordown='#53c156', colorup='#ff1717')
        axes_ls[i].axes.get_yaxis().set_visible(True)
        axes_ls[i].axes.get_xaxis().set_visible(True)
        axes_ls[i].axes.get_yaxis().set_ticks([])
        axes_ls[i].axes.get_xaxis().set_ticks([])


    fig.savefig(filename)
    # plt.show()
    plt.close()