import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, down_sample, in_channel, out_channel, kernel_size):
        super().__init__()

        self.module = torch.nn.Sequential(
            torch.nn.Conv1d(in_channel, out_channel, kernel_size, down_sample, kernel_size // 2),
            torch.nn.PReLU(out_channel),
            torch.nn.Conv1d(out_channel, out_channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(out_channel)
        )

    def forward(self, inputs):
        return self.module(inputs)


class Decoder(torch.nn.Module):
    def __init__(self, channel, kernel_size):
        super().__init__()

        self.module1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel * 2, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel)
        )
        self.module2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, channel, kernel_size, 1, kernel_size // 2),
            torch.nn.PReLU(channel)
        )

    def forward(self, inputs):
        x = F.pad(inputs, (0, 1), value=0)
        x = self.module1(x)
        return self.module2(x)[:, :, :-1]


class UpSampler(torch.nn.Module):
    def __init__(self, up_sampling, channel, kernel_size, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = channel // 2
        self.module = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(channel, out_channel, kernel_size, up_sampling, kernel_size // 2),
            torch.nn.PReLU(out_channel)
        )

    def forward(self, inputs):
        x = F.pad(inputs, (0, 1), value=0)
        return self.module(x)[:, :, :-1]


class Denoiser(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.rank = rank
        self.input = Encoder(2, 1, 16, 5).to(device)
        self.encoder = []
        for i in range(self.rank):
            self.encoder.append(Encoder(2, pow(2, i + 4), pow(2, i + 5), 5).to(device))
        self.up_sampling = []
        for i in range(self.rank, 0, -1):
            self.up_sampling.append(UpSampler(2, pow(2, i + 4), 5).to(device))
        self.decoder = []
        for i in range(self.rank, 0, -1):
            self.decoder.append(Decoder(pow(2, i + 3), 5).to(device))
        self.output_up = UpSampler(2, 16, 5, out_channel=1).to(device)
        self.output = Encoder(1, 2, 1, 5).to(device)
        self.masking = torch.nn.Tanh()

    def forward(self, inputs):
        channel_inputs = torch.unsqueeze(inputs, dim=1)
        e = [self.input(channel_inputs)]
        for i in range(self.rank):
            e.append(self.encoder[i](e[i]))
        up = []
        d = []
        for i in range(self.rank):
            up.append(self.up_sampling[i](e[self.rank - i]))
            d.append(self.decoder[i](torch.cat((e[self.rank - i - 1], up[i]), dim=1)))
        output = self.output_up(d[-1])
        output = self.output(torch.cat((channel_inputs, output), dim=1))
        output = torch.squeeze(output, dim=1)
        output = self.masking(output) * 1.8
        output *= inputs
        return output


