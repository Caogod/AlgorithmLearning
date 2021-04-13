# ********************************
# @auther: Cao Zhengjie
# @date: 2021-4-9
# ********************************

import numpy as np
from scipy import fft,fftpack

def get_feature_freq(data,sample_freq,topN):
    N = len(data)
    ffted_data = fft(data)
    f = np.abs(ffted_data) / N

    freq = fftpack.fftfreq(N, sample_freq)
    s = sorted(ffted_data[1:int(N)/2], reverse=True)

    ret_r = []
    for i in range(topN):
        t = s[i]
        idx = f[:int(len(f)/2)].tolist().index(t)
        ret_r.append((t, freq[idx]))
    return f, freq, ret_r


def find_corr(a,v, type='Pearson'):
    N = len(v)
    n = len(a)
    if type == "Pearson":
        from numpy import corrcoef    # Pearson product-moment correlation coefficients.
        coef = []
        for i in range(N - n):
            coef.append(corrcoef(a, v[i:i + n])[0, 1])
        return coef

    elif type == "Cross":
        from numpy import correlate    # Cross-correlation of two 1-dimensional sequences.
        corr = []
        for i in range(N-n):
            corr.append(correlate(a, v[i:i+n], 'valid').item())
        return corr


def dataset_auto_label(data_set, corr_lsit, seperate_threshold, slice_len):
    label = []
    train_set = []
    for x,c in enumerate(corr_lsit):
        if x < len(corr_lsit):
            train_set.append(data_set[x:x+6])

        if c > seperate_threshold:
            label.append(1)
        else:
            label.append(0)
    return train_set, label
