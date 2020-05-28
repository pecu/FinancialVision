import matplotlib.pyplot as plt
import numpy as np

from .base import Attack
from .base import call_decorator
from ..utils import softmax
from .. import nprng


class SinglePixelAttack(Attack):
    pass


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
        self, input_or_adv, label=None, unpack=True, r=1.2, p=10.0, d=0, t=10, R=150
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

        Im = a.unperturbed
        Im, LB, UB = normalize(Im)

        cI = a.original_class
        channel_axis = a.channel_axis(batch=False)
        axes = [i for i in range(Im.ndim) if i != channel_axis]
        assert len(axes) == 2
        h = Im.shape[axes[0]]
        w = Im.shape[axes[1]]
        channels = Im.shape[channel_axis]

        def random_locations():
            return np.array([[i, i] for i in range(10)])

        def pert(Ii, p, x, y):
            '''
            Ex.
            
            (4, 4)
            Im[location] : [ 0.5        -0.5        -0.47708627  0.47510409]
            p * np.sign(Im[location]) : [ 10. -10. -10.  10.]
            Im[location] : [ 10. -10. -10.  10.]  
            '''
            Im = Ii.copy()
            location = [x, y]
            location.insert(channel_axis, slice(None))
            location = tuple(location)
            Im[location] = p * np.sign(Im[location])
            return Im


        def cyclic(Ibxy):
            r = list(np.random.uniform(0.8, 1.2, 1))[0]
            result = r * Ibxy
            if result <= LB:
                result = Ibxy
            elif result >= UB:
                result = Ibxy
            assert LB <= result <= UB
            return result


        Ii = Im
        PxPy = random_locations()

        # Todo
        init_ls = []
        for channel in range(channels):
            tmp_ls = []
            for x, y in PxPy:
                value = Ii[x, y, channel].copy()
                tmp_ls.append(value)
            init_ls.append(tmp_ls)


        for _run in range(R):

            # Reset
            if (_run + 1) % 10 == 0:
                for channel in range(4):
                    for x, y in PxPy:
                        value = init_ls[channel][x]
                        Ii[x, y, channel] = value

            # Computing the function g using the neighborhood (IMPORTANT: random subset for efficiency)
            PxPy = PxPy[nprng.permutation(len(PxPy))[:128]]
            
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            def score(Its):
                Its = np.stack(Its)
                Its = unnormalize(Its)
                batch_logits, _ = a.forward(Its, strict=False)
                scores = [softmax(logits)[cI] for logits in batch_logits]
                return scores

            scores = score(L)
            indices = np.argsort(scores)[:t]

            PxPy_star = PxPy[indices]

            # Generation of new perturbed input Ii
            for x, y in PxPy_star:
                for b in range(1, channels):
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
            _, is_adv = a.forward_one(Ii)
            if is_adv:
                return

            Ii, LB, UB = normalize(Ii)
