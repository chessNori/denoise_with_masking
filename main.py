import custom_datasets
import models
import personal
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import numpy as np

data_load = False
train_u_bool = False
evaluate = True

n_fft = 512
EPOCHS = 10
batch_size = 8
speech_threshold = 0.00001

if data_load:
    load_path = '../datasets/libriSpeech/train-clean-100'
    file_name = custom_datasets.file_path(load_path, 250000, 'libri')
    print('number of lisf:', len(file_name))

    noise1, __sr__ = librosa.load('../datasets/demand/DKITCHEN/ch01.wav', sr=16000)
    noise2, __sr__ = librosa.load('../datasets/demand/NRIVER/ch01.wav', sr=16000)
    noise3, __sr__ = librosa.load('../datasets/demand/NFIELD/ch01.wav', sr=16000)
    noise4, __sr__ = librosa.load('../datasets/demand/OOFFICE/ch01.wav', sr=16000)
    noise1 = np.expand_dims(noise1[2500:2500 + 512 * 300], 0)
    noise2 = np.expand_dims(noise2[2500:2500 + 512 * 300], 0)
    noise3 = np.expand_dims(noise3[:512 * 300], 0)
    noise4 = np.expand_dims(noise4[2820:2820 + 512 * 300], 0)
    noise_pack = np.concatenate((noise1, noise2, noise3, noise4), axis=0)
    # noise_pack = np.concatenate((noise_pack, np.zeros((4, 256))), axis=1)

    for i in range(5500):
        wave, __sr__ = librosa.load(file_name[i], sr=16000)
        wave = wave[:512 * 300]  # y_data

        wave_temp = np.copy(wave)
        wave_temp = np.concatenate((wave_temp, np.zeros(256)), axis=0)
        wave_temp = torch.Tensor(wave_temp)
        wave_spec = torch.stft(wave_temp, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft,
                               center=False, return_complex=True).numpy()
        mag, phase = librosa.magphase(wave_spec)
        mag = np.log10(mag + 1.0e-7)
        t_mask, f_mask = custom_datasets.make_mask(mag)  # y_data_time, y_data_freq

        wave_temp = np.copy(wave)
        scale1 = personal.adjust_snr(wave_temp, noise1, 5)
        scale2 = personal.adjust_snr(wave_temp, noise2, 5)
        scale3 = personal.adjust_snr(wave_temp, noise3, 5)
        scale4 = personal.adjust_snr(wave_temp, noise4, 5)
        scale_pack = np.array([scale1, scale2, scale3, scale4])
        scale_pack = np.expand_dims(scale_pack, 1)

        wave_temp = np.expand_dims(wave_temp, 0)
        wave_temp = np.concatenate((wave_temp, wave_temp, wave_temp, wave_temp), 0)
        wave_temp += noise_pack * scale_pack  # x_data

        if i < 5000:
            save_path = '../datasets/libriCustom/train/'
            file_index = i
            if i % 1000 == 0:
                print('#', end='')
        else:
            if i == 5000:
                print('Making training dataset is done!')
            save_path = '../datasets/libriCustom/test/'
            file_index = i - 5000
            if i % 100 == 0:
                print('#', end='')

        for j in range(wave_temp.shape[0]):
            sf.write(save_path + 'y_data/' + str(file_index * 4 + j) + '.wav', wave, int(__sr__))  # y_data
            sf.write(save_path + 'x_data/' + str(file_index * 4 + j) + '.wav', wave_temp[j], int(__sr__))  # x_data
            np.save(save_path + 'time_mask/' + str(file_index * 4 + j) + '.npy', t_mask)
            np.save(save_path + 'frequency_mask/' + str(file_index * 4 + j) + '.npy', f_mask)
    print('Making test dataset is done!')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

dataset = custom_datasets.CustomDataset(train=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
dataset_test = custom_datasets.CustomDataset(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

for x, y in dataloader:
    print(x.shape, y.shape)
    break

_model = models.UNET(n_fft=n_fft).to(device)

_optimizer = torch.optim.Adam(_model.parameters(), lr=1e-4)
_loss_fn = torch.nn.MSELoss()


def train_u(data_loader, model,loss_fn, optimizer):
    model.train()
    train_loss = 0.
    for batch_idx, (x_data, y_data) in enumerate(data_loader):
        x_data, y_data = x_data.to(device), y_data.to(device)
        pred = model(x_data)
        cost = loss_fn(pred, y_data)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item()

    train_loss /= len(data_loader) * batch_size
    print('Train Error: {:.6f}'.format(train_loss), end='')


def test_u(data_loader, model, loss_fn):
    test_loss = 0.
    with torch.no_grad():
        for x_data, y_data in data_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            pred = model(x_data)
            cost = loss_fn(pred, y_data)
            test_loss += cost.item()

    test_loss /= len(data_loader) * batch_size
    print(f" // Test Error: {test_loss:>5f}\n")


def evaluate_u(data_loader, denoise_func):  # evaluate
    eval_snr = 0.
    with torch.no_grad():
        for batch_idx, (x_data, y_data) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            pred = denoise_func(x_data)
            bypass = denoise_func(y_data)

            pred, x_data, y_data, bypass = pred.to('cpu'), x_data.to('cpu'), y_data.to('cpu'), bypass.to('cpu')
            pred, x_data, y_data, bypass = pred.numpy(), x_data.numpy(), y_data.numpy(), bypass.numpy()
            eval_snr += personal.snr(y_data, pred) / 32
            if batch_idx == 0:
                for k in range(4):
                    index = k
                    if k == 3:
                        index += 4

                    y_max = np.max(y_data[index])
                    pred_max = np.max(pred[index])
                    eval_temp = np.copy(pred[index])
                    bypass_max = np.max(bypass[index])
                    bypass_temp = np.copy(bypass[index])
                    window_temp = 0.0
                    for num, sample in enumerate(eval_temp):
                        if sample * sample < speech_threshold:  # * 0
                            if window_temp > 0.5:
                                window_temp -= 0.01
                                eval_temp[num] *= window_temp
                            else:
                                eval_temp[num] *= 0.0
                                window_temp = 0.0
                        else:  # * 1
                            if window_temp < 0.5:
                                window_temp += 0.01
                                eval_temp[num] *= window_temp
                            else:
                                eval_temp[num] *= 1.0
                                window_temp = 1.0

                    window_temp = 0.0
                    for num, sample in enumerate(bypass_temp):
                        if sample * sample < speech_threshold:  # * 0
                            if window_temp > 0.5:
                                window_temp -= 0.01
                                bypass_temp[num] *= window_temp
                            else:
                                bypass_temp[num] *= 0.0
                                window_temp = 0.0
                        else:  # * 1
                            if window_temp < 0.5:
                                window_temp += 0.01
                                bypass_temp[num] *= window_temp
                            else:
                                bypass_temp[num] *= 1.0
                                window_temp = 1.0

                    sf.write('./test_files/denoise_eval' + str(index) + '.wav',
                             eval_temp * y_max / pred_max, 16000)
                    sf.write('./test_files/denoise_bypass' + str(index) + '.wav',
                             bypass_temp * y_max / bypass_max, 16000)
                    sf.write('./test_files/denoise_x' + str(index) + '.wav', x_data[index], 16000)
                    sf.write('./test_files/denoise_y' + str(index) + '.wav', y_data[index], 16000)
        eval_snr /= len(data_loader)
        print('Final Model SNR: ', eval_snr, 'dB')


if train_u_bool:
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_u(dataloader, _model, _loss_fn, _optimizer)
        test_u(dataloader_test, _model, _loss_fn)

    torch.save(_model.state_dict(), './saved_models/giant_model5.pt')

    print("Done!")

if evaluate:
    if not train_u_bool:
        state_dict = torch.load('./saved_models/giant_model4.pt', map_location=device)
        _model.load_state_dict(state_dict)

    evaluate_u(dataloader_test, _model)
    print("Done!")
