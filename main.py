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
frame_size = n_fft * 256
EPOCHS = 10000
batch_size = 16
valid_per = 0.1
learning_rate = 0.001

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

_model = models.Denoiser(n_fft).to(device)
_optimizer = torch.optim.AdamW(_model.parameters(), learning_rate)
# last_epoch = -1

_checkpoint = torch.load('./saved_models/giant_model_final_loss.pt', map_location=device)
_model.load_state_dict(_checkpoint['model_state_dict'])
_optimizer.load_state_dict(_checkpoint['optimizer_state_dict'])

last_epoch = _checkpoint['epoch']
_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=15, gamma=0.99, last_epoch=last_epoch)
print("Check Point Epoch:", last_epoch)


def _loss_fn(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def train_u(data_loader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0.
    for x_data, y_data, x_phase, y_phase in data_loader:
        x_data, y_data = x_data.to(device, non_blocking=True), y_data.to(device, non_blocking=True)
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
    model.eval()
    valid_loss = 0.
    valid_pesq = 0.
    exception = 0

    for x_data, y_data, x_phase, y_phase in data_loader:
        with torch.no_grad():
            x_data, y_data = x_data.to(device, non_blocking=True), y_data.to(device, non_blocking=True)
            pred = model(x_data)
            cost = loss_fn(pred, y_data)
            valid_loss += cost.item()
        pred, y_data = pred.to('cpu'), y_data.to('cpu')
        pred = personal.torch_onesided_istft(pred, x_phase, frame_size, n_fft)
        y_data = personal.torch_onesided_istft(y_data, y_phase, frame_size, n_fft)
        pred, y_data = pred.numpy(), y_data.numpy()
        for i in range(x_data.shape[0]):
            try:
                valid_pesq += pesq(16000, y_data[i], pred[i], 'wb')
            except:
                exception += 1

    valid_loss /= len(dataset_valid)
    valid_pesq /= (len(dataset_valid) - exception)

    print(f" // Validation Error: {valid_loss:>8f} // Validation PESQ: {valid_pesq:>8f}")

    return valid_loss, valid_pesq


def test_u(data_loader, denoise_func):  # batch_size = 1
    denoise_func.eval()
    res_list = [['File Name', 'SNR', 'PESQ']]
    list_temp = [None] * 3
    padding = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    padding = torch.sin(padding[1:n_fft//2 + 1])
    with torch.no_grad():
        for x_data, y_data, x_phase, y_phase, wave_length, file_name in data_loader:
            x_data, x_phase = x_data.to(device, non_blocking=True), x_phase.to(device, non_blocking=True)
            pred = denoise_func(x_data)

            pred = personal.torch_onesided_istft(pred, x_phase, wave_length, n_fft)[0]
            y_wave = personal.torch_onesided_istft(y_data, y_phase, wave_length, n_fft)[0]

            pred = pred.to('cpu')
            pred[:n_fft // 2] *= padding
            y_wave[:n_fft // 2] *= padding
            pred, y_wave = pred.numpy(), y_wave.numpy()
            list_temp[0] = file_name[0].split('\\')[-1]
            list_temp[1] = personal.snr(y_wave, pred)
            list_temp[2] = pesq(16000, y_wave, pred, 'wb')
            res_list.append(list_temp.copy())

            sf.write('./test_files/eval/' + file_name[0].split('\\')[-1], pred, 16000)
            sf.write('./test_files/y/' + file_name[0].split('\\')[-1], y_wave, 16000)
    return res_list


if train_u_bool:
    loss_list = [['train_loss'], ['validation_loss'], ['validation_pesq']]
    max_pesq = -1.0
    min_loss = 100.0
    for epoch in range(last_epoch + 1, EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss_temp = train_u(dataloader, _model, _loss_fn, _optimizer)
        valid_loss_temp, valid_pesq_temp = valid_u(dataloader_valid, _model, _loss_fn)
        _scheduler.step()
        if valid_loss_temp < min_loss:
            min_loss = valid_loss_temp
            torch.save({'model_state_dict': _model.state_dict(),
                        'optimizer_state_dict': _optimizer.state_dict(),
                        'epoch': epoch}, './saved_models/giant_model_final_loss.pt')
        if valid_pesq_temp > max_pesq:
            max_pesq = valid_pesq_temp
            torch.save({'model_state_dict': _model.state_dict(),
                        'optimizer_state_dict': _optimizer.state_dict(),
                        'epoch': epoch}, './saved_models/giant_model_final_pesq.pt')
        loss_list[0].append(train_loss_temp)
        loss_list[1].append(valid_loss_temp)
        loss_list[2].append(valid_pesq_temp)
        f = open('loss.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(loss_list[0])
        writer.writerow(loss_list[1])
        writer.writerow(loss_list[2])
        f.close()

    print("Done!")

if evaluate:
    if not train_u_bool:
        checkpoint = torch.load('./saved_models/giant_model_final_pesq.pt', map_location=device)
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
