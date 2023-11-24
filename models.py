import torch
import torch.nn.functional as F


class UNET(torch.nn.Module):
    def __init__(self, n_fft):
        super(UNET, self).__init__()
        self.n_fft = n_fft

        self.encoder1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.encoder2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.encoder3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.latent = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.up_sampling1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU()
        )

        self.decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.up_sampling2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU()
        )
        self.decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.up_sampling3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 2), (1, 2)),
            torch.nn.PReLU()
        )
        self.decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(32, 32, (3, 5), (1, 1), (1, 2)),
            torch.nn.PReLU()
        )

        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, (3, 5), (1, 1), (1, 2)),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        phase = torch.stft(F.pad(inputs, (0, self.n_fft), 'constant', 0),
                           n_fft=self.n_fft, hop_length=self.n_fft//2, win_length=self.n_fft,
                           center=False, return_complex=False)
        mag = torch.sqrt(torch.pow(phase[:, :, :, 0], 2.0) + torch.pow(phase[:, :, :, 1], 2.0))
        phase[:, :, :, 0] /= mag
        phase[:, :, :, 1] /= mag
        mag_db = torch.log10(mag + 1.0e-7)
        mag_db = torch.transpose(mag_db, 1, 2)
        mag_db = torch.unsqueeze(mag_db, 1)  # (32, 1, 201, 257)

        e1 = self.encoder1(mag_db)  # (32, 16, 201, 257)
        e2 = self.encoder2(e1)  # (32, 16, 201, 129)
        e3 = self.encoder3(e2)

        l1 = self.latent(e3)  # (32, 16, 201, 65)

        u1 = self.up_sampling1(l1)  # (32, 16, 201, 129)
        d1 = self.decoder1(torch.cat((e3, u1), 1))  # (32, 16, 201, 129)
        u2 = self.up_sampling2(d1)  # (32, 16, 201, 257)
        d2 = self.decoder2(torch.cat((e2, u2), 1))  # (32, 16, 201, 257)
        u3 = self.up_sampling3(d2)
        d3 = self.decoder3(torch.cat((e1, u3), 1))
        d3 = self.output(d3)  # (32, 1, 201, 257)

        freq_mask = torch.squeeze(d3)  # (32, 201, 257)

        x = masking(inputs, freq_mask)

        return x


def masking(inputs, mask, n_fft=512):
    sine_window = torch.Tensor(range(n_fft + 2)) * (4.0 * torch.atan(torch.Tensor([1.0]))) / (n_fft + 2)
    sine_window = torch.sin(sine_window[1:-1])
    sine_window = sine_window.to('cuda')
    spec = torch.stft(F.pad(inputs, (0, n_fft), 'constant', 0),
                      window=sine_window,
                      n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft,
                      center=False, return_complex=True)
    relu = torch.nn.ReLU()
    p_mask = relu(mask)
    spec *= torch.transpose(p_mask, 1, 2).type(torch.complex64)
    res = torch.istft(spec, n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft,
                      window=sine_window,
                      center=False, return_complex=False)

    return res[:, :n_fft * 300]