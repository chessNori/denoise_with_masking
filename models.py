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
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2

        self.up_sampler = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, channel, kernel_size + 1, 2, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel * 2, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

    def forward(self, skip, inputs):
        x = self.up_sampler(inputs)
        return self.decoder1(torch.cat((skip, x), dim=1))


class Denoiser(torch.nn.Module):
    def __init__(self, rank, n_fft):
        super().__init__()
        self.rank = rank
        self.n_fft = n_fft

        self.encoder = torch.nn.ModuleList([Encoder(5, n_fft) for _ in range(rank)])
        self.decoder = torch.nn.ModuleList([Decoder(5, n_fft) for _ in range(rank)])
        self.output = torch.nn.Conv1d(n_fft // 2, n_fft // 2, 3, 1, 1)
        self.output_act = torch.nn.Sigmoid()

    def forward(self, inputs):
        e = [inputs]
        for i in range(self.rank):
            e.append(self.encoder[i](e[i]))
        x = e[-1]
        for i in range(self.rank - 1, -1, -1):
            x = self.decoder[i](e[i], x)
        mask = self.output(x)
        mask = self.output_act(mask) * 1.3
        mask = mask * inputs

        return mask
