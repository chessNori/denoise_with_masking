import custom_datasets
import models
import personal
import torch
import soundfile as sf
import numpy as np

train_u_bool = False
evaluate = True

frame_size = 512 * 200
EPOCHS = 2000
batch_size = 32
speech_threshold = 0.00001
valid_per = 0.08
learning_rate = 0.005

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

if train_u_bool:
    dataset = custom_datasets.CustomTrainSet(frame_size, valid_per)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break

    dataset_valid = custom_datasets.CustomTrainSet(frame_size, valid_per, train=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=batch_size,
                                                   shuffle=False, drop_last=False)
else:
    dataset_test = custom_datasets.CustomTestSet()
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1,
                                                  shuffle=False, drop_last=False)

_model = models.Denoiser(rank=4).to(device)

_optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
_loss_fn = torch.nn.L1Loss()


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
    print(f" // Test Error: {test_loss:>8f}", end='')

    return test_loss


def evaluate_u(data_loader, denoise_func):  # evaluate
    eval_snr = 0.
    with torch.no_grad():
        for batch_idx, (x_data, y_data) in enumerate(data_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            pred = denoise_func(x_data)

            pred, x_data, y_data = pred.to('cpu'), x_data.to('cpu'), y_data.to('cpu')
            pred, x_data, y_data = pred.numpy(), x_data.numpy(), y_data.numpy()
            eval_snr += personal.snr(y_data, pred) / x_data.shape[0]

            for index in range(x_data.shape[0]):
                y_max = np.max(y_data[index])
                pred_max = np.max(pred[index])
                eval_temp = np.copy(pred[index])
                window_temp = 0.0
                for num, sample in enumerate(eval_temp):
                    if pow(sample, 2) < speech_threshold:  # * 0
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

                sf.write('./test_files/eval/denoise' + str(batch_idx) + '_' + str(index) + '.wav',
                         eval_temp * y_max / pred_max, 16000)
                sf.write('./test_files/y/denoise' + str(batch_idx) + '_' + str(index) + '.wav',
                         y_data[index], 16000)
        eval_snr /= len(data_loader)
        print('Final Model SNR: ', eval_snr, 'dB')


if train_u_bool:
    index = 0
    count = 0
    loss_temp = 0.
    min_loss = 100.
    model_temp = None
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_u(dataloader, _model, _loss_fn, _optimizer)
        loss_temp = test_u(dataloader_valid, _model, _loss_fn)
        print(' // Count:', count)
        if epoch > 100 and min_loss > loss_temp:
            min_loss = loss_temp
            model_temp = _model.state_dict()
            count = 0
        else:
            count += 1
            if epoch > 100 and count > 12:
                torch.save(model_temp, './saved_models/giant_model' + str(index) + '_' + str(min_loss) + '.pt')
                index += 1
                count = 0
    torch.save(model_temp, './saved_models/giant_model' + str(index) + '_' + str(min_loss) + '.pt')
    torch.save(_model.state_dict(), './saved_models/giant_model_final.pt')

    print("Done!")

if evaluate:
    if not train_u_bool:
        state_dict = torch.load('./saved_models/giant_model0_0.0004379171067349879.pt', map_location=device)
        _model.load_state_dict(state_dict)

    evaluate_u(dataloader_test, _model)
    print("Done!")
