import torch


class Encoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2
        self.module = torch.nn.Sequential(
            torch.nn.Conv1d(channel, channel, kernel_size, 2, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, 'same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(channel, channel, kernel_size, 1, 'same'),
            torch.nn.LeakyReLU()
        )

    def forward(self, inputs):
        return self.module(inputs)


class Aligner(torch.nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gate_vector = torch.nn.Linear(channel, channel)
        self.input_feature = torch.nn.Linear(channel, channel)
        self.attention = torch.nn.Linear(channel, channel)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs, query):
        x = self.input_feature(torch.transpose(inputs, 1, 2))
        gate = self.gate_vector(torch.transpose(query, 1, 2))
        x = x + gate
        x = self.leaky_relu(x)
        x = self.attention(x)
        x = self.sigmoid(x)
        x = torch.transpose(x, 1, 2)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, kernel_size, n_fft):
        super().__init__()
        channel = n_fft // 2

        self.up_sampler = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, channel, kernel_size + 1, 2, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

        self.gate_attention = Aligner(channel)

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel * 2, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.LeakyReLU()
        )

    def forward(self, skip, inputs):
        x1 = self.up_sampler(inputs)
        attention_score = self.gate_attention(x1, skip)
        x2 = x1 * attention_score  # skip

        return self.decoder(torch.cat((x1, x2), dim=1))


class Denoiser(torch.nn.Module):
    def __init__(self, rank, n_fft):
        super().__init__()
        self.rank = rank
        self.n_fft = n_fft

        self.encoder = torch.nn.ModuleList([Encoder(5, n_fft) for _ in range(rank)])
        self.decoder = torch.nn.ModuleList([Decoder(5, n_fft) for _ in range(rank)])
        self.output = torch.nn.Conv1d(n_fft // 2, n_fft // 2, 3, 1, 'same')
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
