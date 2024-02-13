import torch


class Encoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2
        self.module = torch.nn.Sequential(
            torch.nn.Conv1d(channel, channel, kernel_size, 2, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

    def forward(self, inputs):
        return self.module(inputs)


class Decoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft, nn_skip=False):
        super().__init__()
        channel = n_fft // 2
        self.nn_skip = nn_skip
        input_rank = 4 if nn_skip else 3

        self.up_sampler = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, channel, kernel_size + 1, 2, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel * input_rank, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

    def forward(self, skip, gru_skip, inputs):
        x = self.up_sampler(inputs)
        x = torch.cat((skip, gru_skip, x), dim=1) if self.nn_skip else torch.cat((skip, x), dim=1)
        return self.decoder(x)


class Denoiser(torch.nn.Module):
    def __init__(self, rank, n_fft):
        super().__init__()
        self.rank = rank
        self.n_fft = n_fft

        self.encoder = torch.nn.ModuleList([Encoder(5, n_fft) for _ in range(rank)])
        self.decoder = torch.nn.ModuleList([Decoder(5, n_fft) for _ in range(rank)])
        self.gru = torch.nn.ModuleList([
            torch.nn.GRU(n_fft // 2, n_fft // 2, num_layers=2,
                         batch_first=True, bidirectional=True) for _ in range(rank)])
        self.output = torch.nn.Conv1d(n_fft // 2, n_fft // 2, 5, 1, 2)
        self.output_act = torch.nn.Sigmoid()

    def forward(self, inputs):
        e = [inputs]
        for i in range(self.rank):
            e.append(self.encoder[i](e[i]))
        x = e[-1]
        for i in range(self.rank - 1, -1, -1):
            gru_skip = torch.transpose(e[i], 1, 2)
            gru_skip = self.gru[i](gru_skip)[0]
            gru_skip = torch.transpose(gru_skip, 1, 2)
            # gru_skip = None
            x = self.decoder[i](gru_skip, None, x)
        x = self.output(x)
        x = self.output_act(x) * 1.6
        x = x * inputs

        return x
