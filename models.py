import torch
import torch.nn.functional as F


class Masking(torch.nn.Module):
    def __init__(self, n_fft):
        super(Masking, self).__init__()
        self.n_fft = n_fft
        self.freq_mask = torch.nn.Sequential(
            torch.nn.Dropout(0.15),
            torch.nn.Conv2d(1, 32, (5, 15), padding=(2, 7)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, (3, 15), padding=(1, 7)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 8, (3, 15), padding=(1, 7)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 1, (3, 15), padding=(1, 7)),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(start_dim=1, end_dim=2),  # (B, 199, 257)
            torch.nn.Linear(257, 257),
            torch.nn.Tanh()
        )

    def forward(self, inputs):
        phase = torch.stft(F.pad(inputs, (0, 256), 'constant', 0),
                           n_fft=self.n_fft, hop_length=self.n_fft//2, win_length=self.n_fft,
                           center=False, return_complex=False)
        mag = torch.sqrt(torch.pow(phase[:, :, :, 0], 2.0) + torch.pow(phase[:, :, :, 1], 2.0))
        phase[:, :, :, 0] /= mag
        phase[:, :, :, 1] /= mag
        mag_db = torch.log10(mag + 1.0e-7)
        mag_db = torch.transpose(mag_db, 1, 2)
        mag_db = torch.unsqueeze(mag_db, 1)
        freq_mask = self.freq_mask(mag_db)  # (16, 199, 257)

        return freq_mask


def masking(inputs, mask, n_fft=512):
    sine_window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    sine_window = torch.sin(sine_window[1:-1])
    sine_window = sine_window.to('cuda')
    spec = torch.stft(F.pad(inputs, (0, 256), 'constant', 0),
                      window=sine_window,
                      n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft,
                      center=False, return_complex=True)
    relu = torch.nn.ReLU()
    p_mask = relu(mask)
    spec *= torch.transpose(p_mask, 1, 2).type(torch.complex64)
    res = torch.istft(spec, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft,
                      window=sine_window,
                      center=False, return_complex=False)

    return res


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 25, 4, 12, bias=False),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(8),
            torch.nn.Conv1d(8, 16, 25, 4, 12, bias=False),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(16),
            torch.nn.Conv1d(16, 32, 25, 4, 12, bias=False),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(32)
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 16, 24, 4, 10, bias=False),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(16),
            torch.nn.ConvTranspose1d(16, 8, 24, 4, 10, bias=False),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(8),
            torch.nn.ConvTranspose1d(8, 1, 24, 4, 10, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, inputs):
        x = torch.unsqueeze(inputs, 1)
        x = self.decoder(x)
        x = self.encoder(x)
        x = torch.squeeze(x)

        return x
