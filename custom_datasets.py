import os
from glob import glob
import numpy as np
import librosa
import torch
import random

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


class CustomTrainSet(torch.utils.data.Dataset):
    def __init__(self, frame_size, valid_per, train=True, seed=42, path='../datasets/nsdtseaCustom', data_sr=16000):
        super().__init__()
        random.seed(seed)

        x_data_path = path + '/noisy_trainset_container/x.wav'
        y_data_path = path + '/clean_trainset_container/y.wav'

        x_data_temp = librosa.load(x_data_path, sr=data_sr, dtype='float32')[0]
        y_data_temp = librosa.load(y_data_path, sr=data_sr, dtype='float32')[0]
        length_temp = frame_size - (x_data_temp.shape[0] % frame_size)
        x_data = torch.zeros(x_data_temp.shape[0] + length_temp, dtype=torch.float32)
        y_data = torch.zeros(y_data_temp.shape[0] + length_temp, dtype=torch.float32)

        x_data[:-length_temp] += x_data_temp
        y_data[:-length_temp] += y_data_temp

        x_data = x_data.reshape(-1, frame_size)
        y_data = y_data.reshape(-1, frame_size)

        valid_idx_list = random.sample(range(x_data.shape[0]), round(x_data.shape[0] * valid_per))
        train_idx_list = list(set(range(x_data.shape[0])) - set(valid_idx_list))
        idx_list = train_idx_list if train else valid_idx_list

        self.x_data, self.y_data = x_data[idx_list], y_data[idx_list]

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class CustomTestSet(torch.utils.data.Dataset):
    def __init__(self, path='../datasets/nsdtseaCustom', data_sr=16000):
        super().__init__()
        self.data_sr = data_sr
        x_data_path = path + '/noisy_testset_wav'
        y_data_path = path + '/clean_testset_wav'

        self.x_data_list = sorted(glob(x_data_path + '/*.wav'))  # 824 data, max length: 156302
        self.y_data_list = sorted(glob(y_data_path + '/*.wav'))

    def __len__(self):
        return len(self.x_data_list)

    def __getitem__(self, idx):
        x_data_temp = librosa.load(self.x_data_list[idx], sr=self.data_sr)[0]
        y_data_temp = librosa.load(self.y_data_list[idx], sr=self.data_sr)[0]

        x_data = None
        y_data = None
        mod = len(x_data_temp) % 64

        if mod != 0:
            x_data = torch.zeros(len(x_data_temp) + 64 - mod)
            y_data = torch.zeros(len(y_data_temp) + 64 - mod)
            x_data[:len(x_data_temp)] += x_data_temp
            y_data[:len(y_data_temp)] += y_data_temp
        else:
            x_data = torch.Tensor(x_data_temp)
            y_data = torch.Tensor(y_data_temp)

        return x_data, y_data
