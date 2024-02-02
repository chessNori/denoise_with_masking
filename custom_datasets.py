from glob import glob
import librosa
import torch
import random
import personal
import torch.utils.data as D


class CustomTrainSet(D.Dataset):
    def __init__(self, n_fft, frame_size, valid_per, train=True, seed=42, path='../datasets/nsdtseaCustom', data_sr=16000):
        super().__init__()
        random.seed(seed)
        window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
        window = torch.sin(window[1:-1])
        frame_window = torch.ones(frame_size)
        frame_window[:n_fft // 2] = window[:n_fft // 2]
        frame_window[(-1) * (n_fft // 2):] = window[n_fft // 2:]


        x_data_path = path + '/noisy_trainset_container/x.wav'
        y_data_path = path + '/clean_trainset_container/y.wav'

        x_data_temp = librosa.load(x_data_path, sr=data_sr, dtype='float32')[0]
        y_data_temp = librosa.load(y_data_path, sr=data_sr, dtype='float32')[0]
        length_temp = frame_size - (x_data_temp.shape[0] % frame_size)
        x_data = torch.zeros(x_data_temp.shape[0] + length_temp, dtype=torch.float32)
        y_data = torch.zeros(y_data_temp.shape[0] + length_temp, dtype=torch.float32)

        x_data[:-length_temp] += x_data_temp
        y_data[:-length_temp] += y_data_temp

        x_data = x_data.reshape(-1, frame_size) * frame_window
        y_data = y_data.reshape(-1, frame_size) * frame_window

        valid_idx_list = random.sample(range(x_data.shape[0]), round(x_data.shape[0] * valid_per))
        train_idx_list = list(set(range(x_data.shape[0])) - set(valid_idx_list))
        idx_list = train_idx_list if train else valid_idx_list

        x_data_mag, x_data_phase = personal.torch_stft_magphase(x_data, n_fft)

        self.x_data, self.y_data = x_data_mag[idx_list], y_data[idx_list]
        self.x_data_phase = x_data_phase[idx_list]

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.x_data_phase[idx]


class CustomTestSet(D.Dataset):
    def __init__(self, n_fft, frame_size, path='../datasets/nsdtseaCustom', data_sr=16000):
        super().__init__()
        self.n_fft = n_fft
        self.data_sr = data_sr
        self.frame_size = frame_size
        x_data_path = path + '/noisy_testset_wav'
        y_data_path = path + '/clean_testset_wav'

        self.x_data_list = sorted(glob(x_data_path + '/*.wav'))  # 824 data, max length: 156302
        self.y_data_list = sorted(glob(y_data_path + '/*.wav'))

    def __len__(self):
        return len(self.x_data_list)

    def __getitem__(self, idx):
        x_data_temp = librosa.load(self.x_data_list[idx], sr=self.data_sr)[0]
        y_data_temp = librosa.load(self.y_data_list[idx], sr=self.data_sr)[0]

        file_length = len(x_data_temp)

        data = None
        mod = len(x_data_temp) % self.frame_size
        if mod != 0:
            data = torch.zeros((2, len(x_data_temp) + self.frame_size - mod))
            data[0, :len(x_data_temp)] += x_data_temp
            data[1, :len(y_data_temp)] += y_data_temp
        else:
            data = torch.Tensor(x_data_temp)

        data_mag, data_phase = personal.torch_stft_magphase(data, self.n_fft)

        return data_mag[0], data_mag[1], data_phase[0], data_phase[1], file_length, self.x_data_list[idx]
