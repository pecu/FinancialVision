import numpy as np

def gaf_encode(ts):
    rescaled_ts = np.zeros(len(ts), float)
    min_ts, max_ts = np.min(ts), np.max(ts)
    diff = max_ts - min_ts
    if diff != 0:
        rescaled_ts = (ts - min_ts) / diff
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))
    # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
    this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)

    return this_gam
            
