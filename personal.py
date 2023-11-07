import numpy as np


def snr(signal1, signal2):
    """
    Calculate SNR between inputs
    :param signal1: (ndarray) Signal
    :param signal2: (ndarray) Signal + Noise
    :return: SNR(dB)
    """

    reg = np.max(signal1, axis=-1) / np.max(signal2, axis=-1)
    reg = np.expand_dims(reg, axis=-1)
    sum_noise = signal1 - signal2 * reg
    sum_original = np.power(np.abs(signal1), 2)
    sum_noise = np.power(np.abs(sum_noise), 2)
    sum_original = np.sum(sum_original, axis=-1)
    sum_noise = np.sum(sum_noise, axis=-1)
    res = np.log10(sum_original/sum_noise) * 10
    res = np.sum(res)

    return res


def adjust_snr(target, noise, db):  # Because of abs, it didn't return good scale value. We need bug fix
    sum_original = np.power(np.abs(target), 2)
    sum_noise = np.power(np.abs(noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    sum_original = np.log10(sum_original)
    sum_noise = np.log10(sum_noise)
    scale = np.power(10, (sum_original-sum_noise)/2-(db/20))
    # SNR = 10 * log(power of signal(S)/power of noise(N))
    # SNR = 10 * (log(S) - log(N) - 2 log(noise scale))
    # log(noise scale) = (log(S) - log(N))/2 - SNR/20

    return scale
