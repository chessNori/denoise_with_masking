import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2
        self.module = torch.nn.Sequential(
            torch.nn.Conv1d(channel, channel, kernel_size, 2, kernel_size // 2),
            torch.nn.PReLU(channel),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel)
        )

    def forward(self, inputs):
        return self.module(inputs)


class Decoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2

        self.up_sampler = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 2, kernel_size // 2),
            torch.nn.PReLU(channel)
        )

        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel * 2, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel)

        )

    def forward(self, skip, inputs):
        x = self.up_sampler(inputs)
        return self.decoder1(torch.cat((skip, x), dim=1))


class Denoiser(torch.nn.Module):
    def __init__(self, rank, n_fft):
        super().__init__()
        self.rank = rank
        self.n_fft = n_fft

        self.encoder = torch.nn.ModuleList([Encoder(3, n_fft) for _ in range(rank)])
        self.decoder = torch.nn.ModuleList([Decoder(3, n_fft) for _ in range(rank)])
        self.output = torch.nn.Conv1d(n_fft // 2, n_fft // 2, 3, 1, 1)
        self.output_act = torch.nn.Sigmoid()

    def forward(self, inputs):
        sine_window = torch.Tensor(range(self.n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (self.n_fft + 2)
        sine_window = torch.sin(sine_window[1:-1]).to(inputs.device)
        spectrogram = torch.stft(F.pad(inputs, (0, self.n_fft), 'constant', 0),
                                 window=sine_window, n_fft=self.n_fft, hop_length=self.n_fft // 2,
                                 win_length=self.n_fft, center=False, return_complex=True)
        mag = torch.abs(spectrogram[:, 1:])
        mag_db = torch.log10(mag + 1.0e-7)

        e = [mag_db]
        for i in range(self.rank):
            e.append(self.encoder[i](e[i]))
        x = e[-1]
        for i in range(self.rank - 1, -1, -1):
            x = self.decoder[i](e[i], x)
        mask = self.output(x)
        mask = self.output_act(mask) * 1.2
        mask = F.pad(mask, (0, 0, 1, 0), 'constant', 0)
        spectrogram *= mask.type(torch.complex64)
        x = torch.istft(spectrogram, n_fft=self.n_fft, hop_length=self.n_fft // 2, win_length=self.n_fft,
                        window=sine_window, center=False, return_complex=False)[:, :(-1) * (self.n_fft // 2)]

        return x
