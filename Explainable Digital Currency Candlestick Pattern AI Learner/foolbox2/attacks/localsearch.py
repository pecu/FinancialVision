import matplotlib.pyplot as plt
import numpy as np

from .base import Attack
from .base import call_decorator
from ..utils import softmax
from .. import nprng


class SinglePixelAttack(Attack):
    """Perturbs just a single pixel and sets it to the min or max."""

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True, max_pixels=1000):

        """Perturbs just a single pixel and sets it to the min or max.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        max_pixels : int
            Maximum number of pixels to try.

        """

        a = input_or_adv  # <class 'foolbox.adversarial.Adversarial'>
        del input_or_adv
        del label
        del unpack

        channel_axis = a.channel_axis(batch=False)  # 2

        axes = [i for i in range(a.unperturbed.ndim) if i != channel_axis]  # [0, 1]

        assert len(axes) == 2
        # print(a.unperturbed)               # (32, 32, 4)
        # print(a.unperturbed.shape)         # (32, 32)
        # h = a.unperturbed.shape[axes[0]]   # 32
        # w = a.unperturbed.shape[axes[1]]   # 32

        min_, max_ = a.bounds()              # -1, 1

        # pixels = nprng.permutation(h * w)  # [685 212 489 ... 301 509 759] unique
        # pixels = pixels[:max_pixels]       # max_pixels : 1000
        # for i, pixel in enumerate(pixels):
        #     x = pixel % w
        #     y = pixel // w

        #     location = [x, y]
        #     location.insert(channel_axis, slice(None))
        #     location = tuple(location)

        #     for value in [min_, max_]:
        #         perturbed = a.unperturbed.copy()
        #         perturbed[location] = value

        #         _, is_adv = a.forward_one(perturbed)
        #         if is_adv:
        #             return

        pixels_xy = [(i, i) for i in range(32)]
        for x, y in pixels_xy:

            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)

            for value in [min_, max_]:
                perturbed = a.unperturbed.copy()
                perturbed[location] = value

                _, is_adv = a.forward_one(perturbed)
                if is_adv:
                    return


class LocalSearchAttack(Attack):
    """A black-box attack based on the idea of greedy local search.

    This implementation is based on the algorithm in [1]_.

    References
    ----------
    .. [1] Nina Narodytska, Shiva Prasad Kasiviswanathan, "Simple
           Black-Box Adversarial Perturbations for Deep Networks",
           https://arxiv.org/abs/1612.06299

    """

    @call_decorator
    def __call__(
        self, input_or_adv, num_attacked=None, channel_attacked=None, label=None, unpack=True, r=1.2, p=10.0, d=0, t=10, R=199
        # self, input_or_adv, label=None, unpack=True, r=1.5, p=10.0, d=5, t=5, R=150
    ):

        """A black-box attack based on the idea of greedy local search.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        r : float
            Perturbation parameter that controls the cyclic perturbation;
            must be in [0, 2]
        p : float
            Perturbation parameter that controls the pixel sensitivity
            estimation
        d : int
            The half side length of the neighborhood square
        t : int
            The number of pixels perturbed at each round
        R : int
            An upper bound on the number of iterations

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert 0 <= r <= 2

        if a.target_class is not None:
            return

        def normalize(im):
            min_, max_ = a.bounds()

            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            min_, max_ = a.bounds()

            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        Im = a.unperturbed          # (10, 10, 4)
        Im, LB, UB = normalize(Im)  # normalize to [-0.5, 0.5], LB = -0.5, UB = 0.5

        # (int) ex. 3
        channel_axis = a.channel_axis(batch=False)
        channels = Im.shape[channel_axis]


        def random_locations():
            return np.array([[i, i] for i in num_attacked])


        def cyclic(Ibxy):
            r = list(np.random.uniform(0.8, 1.2, 1))[0]
            # r = list(np.random.uniform(0.9, 1.1, 1))[0]
            # r = list(np.random.uniform(0.95, 1.05, 1))[0]
            # r = list(np.random.uniform(0.85, 1.15, 1))[0]
            # r = list(np.random.uniform(0.4, 1.6, 1))[0]
            # r = list(np.random.uniform(0, 2, 1))[0]
            result = r * Ibxy
            if result <= LB:    # - 0.5
                result = Ibxy
            elif result >= UB:  #   0.5
                result = Ibxy
            assert LB <= result <= UB
            return result


        Ii = Im                    # (10, 10, 4)
        PxPy = random_locations()  # [[0, 0], [1, 1], ..., [9, 9]]

        # Todo
        init_ls = []
        for channel in range(channels):
            tmp_ls = []
            for x, y in PxPy:
                value = Ii[x, y, channel].copy()
                tmp_ls.append(value)
            init_ls.append(tmp_ls)

        # --------- main loop ------------
        for _run in range(R):
            # print('iter : %s' % _run)

            # Reset
            if (_run + 1) % 7 == 0:
                for channel in range(4):
                    PxPy = np.sort(PxPy, axis=0)
                    for idx, (x, y) in enumerate(PxPy):
                        value = init_ls[channel][idx]
                        Ii[x, y, channel] = value

            # Computing the function g using the neighborhood (IMPORTANT: random subset for efficiency)
            PxPy = PxPy[nprng.permutation(len(PxPy))[:128]]

            # Generation of new perturbed input Ii
            for x, y in PxPy:
                for b in channel_attacked:
                    location = [x, y]
                    location.insert(channel_axis, b)
                    location = tuple(location)
                    Ii[location] = cyclic(Ii[location])

            Ii = unnormalize(Ii)

            # Reconstruct GASF
            total_ts = []
            for c in range(4):
                ts = []
                for i in range(10):
                    ts.append(np.cos(np.arccos(Ii[i, i, c]) / 2))
                total_ts.append(ts)

            # New GAF
            Ii = np.zeros((10, 10, 4))            
            for channel in range(4):
                ts_n = total_ts[channel]
                # Arccos
                
                ts_n_arc = np.arccos(ts_n)
                for r in range(10):
                    for c in range(10):
                        Ii[r, c, channel] = np.cos(ts_n_arc[r] + ts_n_arc[c])
            
            # Check whether the perturbed input Ii is an adversarial input
            # (predictions, is_adversarial) < forward_one
            _, is_adv = a.forward_one(Ii)
            if is_adv:
                return None

            Ii, LB, UB = normalize(Ii)

