import custom_datasets
import models
import personal
import torch
import soundfile as sf
import numpy as np
import csv
import torch.nn.functional as F

train_u_bool = False
evaluate = True

n_fft = 512
frame_size = n_fft * 256
EPOCHS = 1000
batch_size = 32
speech_threshold = 0.0001
valid_per = 0.08
learning_rate = 0.0001

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

if train_u_bool:
    dataset = custom_datasets.CustomTrainSet(n_fft, frame_size, valid_per)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break

    dataset_valid = custom_datasets.CustomTrainSet(n_fft, frame_size, valid_per, train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch_size,
                                                   shuffle=False, drop_last=False)
else:
    dataset_test = custom_datasets.CustomTestSet(n_fft, frame_size)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1,
                                                  shuffle=False, drop_last=False)

_model = models.Denoiser(rank=8, n_fft=n_fft, device=device).to(device)
_optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=4, gamma=0.99, last_epoch=-1)


def _loss_fn(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def train_u(data_loader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0.
    for x_data, y_data in data_loader:
        x_data, y_data = x_data.to(device), y_data.to(device)
        pred = model(x_data)
        cost = loss_fn(pred, y_data)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item()

    train_loss /= len(data_loader)
    print('Train Error: {:.6f}'.format(train_loss), end='')


def valid_u(data_loader, model, loss_fn):
    test_loss = 0.
    with torch.no_grad():
        for x_data, y_data in data_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            pred = model(x_data)
            cost = loss_fn(pred, y_data)
            test_loss += cost.item()

    test_loss /= len(data_loader) * batch_size
    print(f" // Test Error: {test_loss:>8f}")

    return test_loss


def test_u(data_loader, denoise_func):  # batch_size = 1
    eval_snr = 0.
    window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    window = torch.sin(window[1:-1]).to(device)
    with (torch.no_grad()):
        for batch_idx, (x_data, y_data, x_phase, y_phase, wave_length) in enumerate(data_loader):
            x_data, y_data, x_phase, y_phase = \
                x_data.to(device), y_data.to(device), x_phase.to(device), y_phase.to(device)
            pred = denoise_func(x_data)

            pred = F.pad(pred, (0, 0, 1, 0), 'constant', 0)
            pred_real = pred * torch.cos(x_phase)
            pred_imag = pred * torch.sin(x_phase)
            pred = torch.complex(pred_real, pred_imag)

            pred = torch.istft(pred, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft,
                               window=window, center=False, return_complex=False)[0, :wave_length]

            y_wave = F.pad(y_data, (0, 0, 1, 0), 'constant', 0)
            y_real = y_wave * torch.cos(y_phase)
            y_imag = y_wave * torch.sin(y_phase)
            y_wave = torch.complex(y_real, y_imag)

            y_wave = torch.istft(y_wave, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft,
                                 window=window, center=False, return_complex=False)[0, :wave_length]

            pred, x_data, y_wave = pred.to('cpu'), x_data.to('cpu'), y_wave.to('cpu')
            pred, x_data, y_wave = pred.numpy(), x_data.numpy(), y_wave.numpy()
            eval_snr += personal.snr(y_wave, pred) / x_data.shape[0]

            # eval_temp = np.copy(pred)
            # window_temp = 0.0
            # for num, sample in enumerate(eval_temp):
            #     if pow(sample, 2) < speech_threshold:  # * 0
            #         if window_temp > 0.5:
            #             window_temp -= 0.01
            #             eval_temp[num] *= window_temp
            #         else:
            #             eval_temp[num] *= 0.0
            #             window_temp = 0.0
            #     else:  # * 1
            #         if window_temp < 0.5:
            #             window_temp += 0.01
            #             eval_temp[num] *= window_temp
            #         else:
            #             eval_temp[num] *= 1.0
            #             window_temp = 1.0

            sf.write('./test_files/eval/denoise' + str(batch_idx) + '.wav',
                     pred, 16000)
            sf.write('./test_files/y/denoise' + str(batch_idx) + '.wav',
                     y_wave, 16000)
        eval_snr /= len(data_loader)
        print('Final Model SNR: ', eval_snr, 'dB')


if train_u_bool:
    loss_list = []
    min_loss = 100.
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_u(dataloader, _model, _loss_fn, _optimizer)
        loss_temp = valid_u(dataloader_valid, _model, _loss_fn)
        _scheduler.step()
        if loss_temp < min_loss:
            min_loss = loss_temp
            torch.save(_model.state_dict(), './saved_models/giant_model_final.pt')
        loss_list.append(loss_temp)
        f = open('loss.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(loss_list)
        f.close()

    print("Done!")

if evaluate:
    if not train_u_bool:
        state_dict = torch.load('./saved_models/giant_model_final.pt', map_location=device)
        _model.load_state_dict(state_dict)

    test_u(dataloader_test, _model)
    print("Done!")
