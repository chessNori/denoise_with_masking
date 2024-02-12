import custom_datasets
import models
import personal
import torch
import torch.utils.data as D
import soundfile as sf
from pesq import pesq
import csv

train_u_bool = False
evaluate = True

n_fft = 512
rank = 7
frame_size = n_fft * pow(2, rank - 1)
EPOCHS = 1000
batch_size = 16
valid_per = 0.1
learning_rate = 0.0003

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


dataset = custom_datasets.CustomTrainSet(n_fft, frame_size, valid_per)\
    if train_u_bool else custom_datasets.CustomTestSet(n_fft, frame_size)

dataloader = D.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)\
    if train_u_bool else D.DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

dataset_valid = custom_datasets.CustomTrainSet(n_fft, frame_size, valid_per, train=False)\
    if train_u_bool else None

dataloader_valid = D.DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False)\
    if train_u_bool else None

for xy in dataloader:
    print(len(dataset), "data:", xy[0].shape, "//", xy[1].shape)
    break

_model = models.Denoiser(rank=rank, n_fft=n_fft).to(device)
_optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=10, gamma=0.99, last_epoch=-1)


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

    train_loss /= len(dataset)

    print('Train Error: {:.6f}'.format(train_loss), end='')

    return train_loss


def valid_u(data_loader, model, loss_fn):
    valid_loss = 0.

    for x_data, y_data in data_loader:
        with torch.no_grad():
            x_data, y_data = x_data.to(device), y_data.to(device)
            pred = model(x_data)
            cost = loss_fn(pred, y_data)
            valid_loss += cost.item()

    valid_loss /= len(dataset_valid)

    print(f" // Validation Error: {valid_loss:>8f}")

    return valid_loss


def test_u(data_loader, denoise_func):  # batch_size = 1
    res_list = [['File Name', 'SNR', 'PESQ']]
    list_temp = [None] * 3
    with torch.no_grad():
        for x_data, y_data, x_phase, y_phase, wave_length, file_name in data_loader:
            x_data = x_data.to(device)
            pred = denoise_func(x_data)[0]

            pred = pred.to('cpu')
            pred = personal.torch_onesided_istft(pred, x_phase, wave_length, n_fft)[0]
            y_data = personal.torch_onesided_istft(y_data, y_phase, wave_length, n_fft)[0]
            pred, y_data = pred.numpy(), y_data.numpy()

            list_temp[0] = file_name[0].split('\\')[-1]
            list_temp[1] = personal.snr(y_data, pred)
            list_temp[2] = pesq(16000, y_data, pred, 'wb')
            res_list.append(list_temp.copy())

            sf.write('./test_files/eval/' + file_name[0].split('\\')[-1], pred, 16000)
            sf.write('./test_files/y/' + file_name[0].split('\\')[-1], y_data, 16000)
    return res_list


if train_u_bool:
    loss_list = [['train_loss'], ['validation_loss']]
    max_pesq = -1.0
    min_loss = 100.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss_temp = train_u(dataloader, _model, _loss_fn, _optimizer)
        valid_loss_temp = valid_u(dataloader_valid, _model, _loss_fn)
        _scheduler.step()
        if valid_loss_temp < min_loss:
            min_loss = valid_loss_temp
            torch.save({'model_state_dict': _model.state_dict(),
                        'optimizer_state_dict': _optimizer.state_dict(),
                        'epoch': epoch}, './saved_models/giant_model_final_loss.pt')

        loss_list[0].append(train_loss_temp)
        loss_list[1].append(valid_loss_temp)
        f = open('loss.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(loss_list[0])
        writer.writerow(loss_list[1])
        f.close()

    print("Done!")

if evaluate:
    if not train_u_bool:
        checkpoint = torch.load('./saved_models/giant_model_final_loss.pt', map_location=device)
        _model.load_state_dict(checkpoint['model_state_dict'])

    test_list = test_u(dataloader, _model)
    eval_snr = 0.
    eval_pesq = 0.
    for i in range(1, len(test_list)):
        eval_snr += test_list[i][1]
        eval_pesq += test_list[i][2]
    eval_snr /= len(dataset)
    eval_pesq /= len(dataset)

    print(f"Final Model SNR: {eval_snr:>8f}dB // Final Model PESQ: {eval_pesq:>8f}")

    f = open('results_pesq.csv', 'w', newline='')
    writer = csv.writer(f)
    for i in range(len(test_list)):
        writer.writerow(test_list[i])
    f.close()
    print("Done!")
