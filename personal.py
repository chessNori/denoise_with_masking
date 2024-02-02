import numpy as np
import torch
import torch.nn.functional as F


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


def reverse_tensor(tensor, dim):
    tensor_size = tensor.shape
    if len(tensor_size) <= dim:
        print("Error: The dim parameter is more than input tensor dimension.")
        print("Input tensor shape:", tensor_size, "// Input dimension:", dim)
        exit()
    tensor_size = tensor_size[dim]
    res_temp = torch.clone(tensor)
    for i in range(tensor_size):
        if dim == 0:
            res_temp[i] = tensor[tensor_size - i - 1]
        elif dim == 1:
            res_temp[:, i] = tensor[:, tensor_size - i - 1]
        elif dim == 2:
            res_temp[:, :, i] = tensor[:, :, tensor_size - i - 1]
        elif dim == 4:
            res_temp[:, :, :, i] = tensor[:, :, :, tensor_size - i - 1]
        else:
            print("Error: The dim parameter is out of range.(0 ~ 4)")
            break

    return res_temp


def torch_stft_magphase(wave, n_fft, dc=False, even_frame=True):
    window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    window = torch.sin(window[1:-1])

    temp = torch.Tensor(wave).type(torch.float32)
    temp = F.pad(temp, (0, n_fft // 2), mode='constant', value=0) if even_frame else wave
    temp = torch.stft(temp, n_fft, n_fft // 2, n_fft, window, False, return_complex=True) if dc\
        else torch.stft(temp, n_fft, n_fft // 2, n_fft, window, False, return_complex=True)[:, 1:]

    mag = torch.abs(temp)
    phase = torch.angle(temp)

    return mag, phase


def torch_onesided_istft(mag, phase, wave_size, n_fft, dc=False):
    """
    Operate pytorch istft with half of spectrum's magnitude.
    :param mag: (tensor) Magnitude of half of spectrum. Shape: (Batch, Magnitude, Time)
    :param phase: (tensor) Phase of half of spectrum. Shape: (Batch, Phase, Time)
    :param wave_size: (int) Length of original wave.
    :param n_fft: (int) N parameter of FFT.
    :param dc: (bool) Whether to use DC bin or not.
    :return: (tensor) Tensor of results of istft operation.
    """
    device = mag.device
    window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    window = torch.sin(window[1:-1]).to(device)

    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    spec = torch.complex(real, imag)[:, 1:] if dc else torch.complex(real, imag)

    temp = torch.zeros(spec.shape[0], n_fft, spec.shape[-1], dtype=torch.complex64).to(device)
    temp[:, 1:n_fft // 2 + 1] += spec
    temp[:, n_fft // 2 + 1:] += torch.conj(reverse_tensor(spec[:, :-1], dim=1))

    res = torch.istft(temp, n_fft, n_fft // 2, n_fft, window, False,
                      onesided=False, return_complex=True)[:, :wave_size].real.to(device)

    return res
