import torch
from torch.nn import GRU, BatchNorm1d, Linear, ReLU, Sequential, Module


class Denoiser(Module):
    def __init__(self, n_fft):
        super().__init__()
        self.batch_norm1 = BatchNorm1d(n_fft // 2)
        self.en_gru1 = GRU(n_fft // 2, n_fft, batch_first=True, bidirectional=True)
        self.en_gru2 = GRU(n_fft * 2, n_fft, batch_first=True, bidirectional=True)
        self.latent = Linear(n_fft * 2, n_fft * 2)
        self.batch_norm2 = BatchNorm1d(n_fft * 2)
        self.de_gru1 = GRU(n_fft * 2, n_fft, batch_first=True, bidirectional=True)
        self.de_gru2 = GRU(n_fft * 2, n_fft, batch_first=True, bidirectional=True)
        self.output = Sequential(Linear(n_fft * 2, n_fft // 2),
                                 ReLU())

    def forward(self, inputs):  # inputs: (B, F, T)
        x = self.batch_norm1(inputs)  # (16, 256, 512)
        x = torch.transpose(x, 1, 2)  # (16, 512, 256)
        x = self.en_gru1(x)[0]  # (16, 512, 1024)
        x = self.en_gru2(x)[0]  # (16, 512, 1024)
        x = self.latent(x)  # (16, 512, 1024)
        x = torch.transpose(x, 1, 2)  # (16, 1024, 512)
        x = self.batch_norm2(x)  # (16, 1024, 512)
        x = torch.transpose(x, 1, 2)  # (16, 512, 1024)
        x = self.de_gru1(x)[0]  # (16, 512, 1024)
        x = self.de_gru2(x)[0]  # (16, 512, 1024)
        x = self.output(x)  # (16, 512, 256)
        x = torch.transpose(x, 1, 2)

        return x
