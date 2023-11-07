import custom_datasets
import models
import personal
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import numpy as np

data_load = False
train_mask = False
train_ae = False
evaluate = True

n_fft = 512
EPOCHS = 50

if data_load:
    load_path = '../datasets/libriSpeech/train-clean-100'
    file_name = custom_datasets.file_path(load_path, 250000, 'libri')
    print('number of lisf:', len(file_name))

    noise1, __sr__ = librosa.load('../datasets/demand/DKITCHEN/ch01.wav', sr=16000)
    noise2, __sr__ = librosa.load('../datasets/demand/NRIVER/ch01.wav', sr=16000)
    noise3, __sr__ = librosa.load('../datasets/demand/NFIELD/ch01.wav', sr=16000)
    noise4, __sr__ = librosa.load('../datasets/demand/OOFFICE/ch01.wav', sr=16000)
    noise1 = np.expand_dims(noise1[2500:2500 + 512 * 100], 0)
    noise2 = np.expand_dims(noise2[2500:2500 + 512 * 100], 0)
    noise3 = np.expand_dims(noise3[:512 * 100], 0)
    noise4 = np.expand_dims(noise4[2820:2820 + 512 * 100], 0)
    noise_pack = np.concatenate((noise1, noise2, noise3, noise4), axis=0)
    # noise_pack = np.concatenate((noise_pack, np.zeros((4, 256))), axis=1)

    for i in range(5500):
        wave, __sr__ = librosa.load(file_name[i], sr=16000)
        wave = wave[:512 * 100]  # y_data

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
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=False)
dataset_test = custom_datasets.CustomDataset(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=32, shuffle=False, drop_last=False)

for x, y, y_f in dataloader:
    print(x.shape, y.shape, y_f.shape)
    break

mask_model = models.Masking(n_fft=n_fft).to(device)

_optimizer_mask = torch.optim.Adam(mask_model.parameters(), lr=1e-4)
_loss_fn_mask = torch.nn.MSELoss()


def train_step(data_loader, model, loss_fn, optimizer):  # mask
    model.train()
    train_loss = 0.
    for batch_idx, (x_data, y_data, y_data_freq) in enumerate(data_loader):
        x_data, y_data_freq = x_data.to(device), y_data_freq.to(device)
        freq_mask = model(x_data)
        cost = loss_fn(freq_mask, y_data_freq)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item()
    train_loss /= len(data_loader)

    print('Train Error: {:.6f}'.format(train_loss), end='')


def test_step(data_loader, model, loss_fn):  # mask
    test_loss = 0.
    with torch.no_grad():
        for x_data, y_data, y_data_freq in data_loader:
            x_data, y_data_freq = x_data.to(device), y_data_freq.to(device)
            freq_mask = model(x_data)
            cost = loss_fn(freq_mask, y_data_freq)
            test_loss += cost.item()
    test_loss /= len(data_loader)
    print(f" // Test Error: {test_loss:>5f}\n")


denoise_model = models.Autoencoder().to(device)

_optimizer_denoise = torch.optim.Adam(denoise_model.parameters(), lr=1e-4)
_loss_fn_denoise = torch.nn.MSELoss()


def train_s(data_loader, model, loss_fn, optimizer, mask_func):  # denoise
    model.train()
    train_loss = 0.
    for batch_idx, (x_data, y_data, y_data_freq) in enumerate(data_loader):
        x_data, y_data, y_data_freq = x_data.to(device), y_data.to(device), y_data_freq.to(device)
        with torch.no_grad():
            freq_mask = mask_func(x_data)  # generate mask
            xx_data = models.masking(x_data, freq_mask)  # masking x_data
            yy_data = models.masking(y_data, y_data_freq)  # masking y_data
        pred = model(xx_data)
        cost = loss_fn(pred, yy_data)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item()
    train_loss /= len(data_loader)

    print('Train Error: {:.6f}'.format(train_loss), end='')


def test_s(data_loader, model, loss_fn, mask_func):  # denoise
    test_loss = 0.
    with torch.no_grad():
        for x_data, y_data, y_data_freq in data_loader:
            x_data, y_data, y_data_freq = x_data.to(device), y_data.to(device), y_data_freq.to(device)
            freq_mask = mask_func(x_data)  # generate mask
            xx_data = models.masking(x_data, freq_mask)  # masking x_data
            yy_data = models.masking(y_data, y_data_freq)  # masking y_data
            pred = model(xx_data)
            cost = loss_fn(pred, yy_data)
            test_loss += cost.item()

    test_loss /= len(data_loader)
    print(f" // Test Error: {test_loss:>5f}\n")


def evaluate_s(data_loader, mask_func, denoise_func):  # evaluate
    eval_snr = 0.
    xx_snr = 0.
    with torch.no_grad():
        for batch_idx, (x_data, y_data, y_data_freq) in enumerate(data_loader):
            x_data = x_data.to(device)
            freq_mask = mask_func(x_data)  # generate mask
            xx_data = models.masking(x_data, freq_mask)  # masking x_data
            pred = denoise_func(xx_data)
            pred = models.masking(pred, freq_mask)  # masking pred

            pred, x_data, xx_data = pred.to('cpu'), x_data.to('cpu'), xx_data.to('cpu')
            pred, x_data, xx_data, y_data = pred.numpy(), x_data.numpy(), xx_data.numpy(), y_data.numpy()
            eval_snr += personal.snr(y_data, pred) / 32
            xx_snr += personal.snr(xx_data, pred) / 32
            if batch_idx == 0:
                for k in range(4):
                    index = k
                    if k == 3:
                        index += 4

                    y_max = np.max(y_data[index])
                    xx_max = np.max(xx_data[index])
                    pred_max = np.max(pred[index])
                    sf.write('./test_files/denoise_eval' + str(index) + '.wav',
                             pred[index] * y_max / pred_max, 16000)
                    sf.write('./test_files/denoise_x' + str(index) + '.wav', x_data[index], 16000)
                    sf.write('./test_files/denoise_xx' + str(index) + '.wav',
                             xx_data[index] * y_max / xx_max, 16000)
                    sf.write('./test_files/denoise_y' + str(index) + '.wav', y_data[index], 16000)
        eval_snr /= len(data_loader)
        xx_snr /= len(data_loader)
        print('Final Model SNR: ', eval_snr, 'dB')
        print('Masking SNR: ', xx_snr, 'dB')


if train_mask:
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_step(dataloader, mask_model, _loss_fn_mask, _optimizer_mask)
        test_step(dataloader_test, mask_model, _loss_fn_mask)

    torch.save(mask_model.state_dict(), './saved_models/mask_model/model1.pt')

    print("Done!")

if train_ae:
    if not train_mask:
        mask_state_dict = torch.load('./saved_models/mask_model/model1.pt', map_location=device)
        mask_model.load_state_dict(mask_state_dict)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_s(dataloader, denoise_model, _loss_fn_denoise, _optimizer_denoise, mask_model)
        test_s(dataloader_test, denoise_model, _loss_fn_denoise, mask_model)

    torch.save(denoise_model.state_dict(),  './saved_models/denoise_model/model1.pt')

    print("Done!")


if evaluate:
    if not train_mask:
        mask_state_dict = torch.load('./saved_models/mask_model/model1.pt', map_location=device)
        mask_model.load_state_dict(mask_state_dict)
    if not train_ae:
        denoise_state_dict = torch.load('./saved_models/denoise_model/model1.pt', map_location=device)
        denoise_model.load_state_dict(denoise_state_dict)

    evaluate_s(dataloader_test, mask_model, denoise_model)
    print("Done!")
