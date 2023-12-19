import os
from glob import glob
import numpy as np
import librosa
import torch


def file_path(basic_path, min_size, file_name):
    segment_name = []
    if file_name[:5] == 'libri':
        speaker_dir = [f.path for f in os.scandir(basic_path) if f.is_dir()]

        chapter_dir = []
        for one_path in speaker_dir:
            chapter_dir += [f.path for f in os.scandir(one_path) if f.is_dir()]

        for one_path in chapter_dir:
            segment_name += glob(one_path + '/*.flac')

    elif file_name[:6] == 'demand':
        noise_dir = [f.path for f in os.scandir(basic_path) if f.is_dir()]

        for one_path in noise_dir:
            segment_name += glob(one_path + '/*.wav')

    else:
        print("Method for this dataset is not ready")
        return -1

    delete_file = []
    for one_path in segment_name:
        if os.stat(one_path).st_size < min_size:
            delete_file.append(one_path)

    for one_path in delete_file:
        segment_name.remove(one_path)  # Delete too small segment

    return segment_name


def make_mask(wave_mag, th_hi=-0.9, th_lo=-0.7, time_cut=15):
    mask = np.zeros_like(wave_mag)
    for i in range(wave_mag.shape[0]):
        for j in range(wave_mag.shape[1]):
            if (wave_mag[i][j] > th_hi) and (i > 128):
                mask[i][j] = 1
            elif (wave_mag[i][j] > th_lo) and (i <= 128):
                mask[i][j] = 1

    time_masking = np.sum(mask, axis=0)

    for i in range(time_masking.shape[0]):
        if time_masking[i] < time_cut:
            mask[:, i] -= mask[:, i]  # time masking
            time_masking[i] = 0
        else:
            time_masking[i] = 1

    mask[0:4, :] = 0  # remove DC

    mask = np.transpose(mask, (1, 0))

    return time_masking, mask


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, path='../datasets/libriCustom', data_sr=16000):
        super(CustomDataset, self).__init__()
        self.path = path
        self.sr = data_sr
        if train:
            self.x_data_path = path + '/noisy_trainset_28spk_wav'
            self.y_data_path = path + '/clean_trainset_28spk_wav'
        else:
            self.x_data_path = path + '/noisy_testset_wav'
            self.y_data_path = path + '/clean_testset_wav'

        self.x_data_list = sorted(glob(self.x_data_path + '/*.wav'))
        self.y_data_list = sorted(glob(self.y_data_path + '/*.wav'))

    def __len__(self):
        return len(self.x_data_list)

    def __getitem__(self, idx):
        x_data, _ = librosa.load(self.x_data_list[idx], sr=self.sr)
        y_data, _ = librosa.load(self.y_data_list[idx], sr=self.sr)
        size = x_data.shape[0]

        if size > 512 * 200:
            x_data = x_data[:512 * 200]
            y_data = y_data[:512 * 200]
        else:
            zeros = np.zeros(512 * 200 - size)
            x_data = np.concatenate((x_data, zeros)).astype('float32')
            y_data = np.concatenate((y_data, zeros)).astype('float32')

        x_data /= np.max(np.abs(x_data))
        y_data /= np.max(np.abs(y_data))

        return x_data, y_data
