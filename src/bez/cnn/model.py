from torch import nn


class ResBlock(nn.Module):
    def __init__(self, d, kernel_size, dilation=3):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(d, d, kernel_size, dilation=dilation, padding=padding)
        self.reg = nn.SiLU()

    def forward(self, x):
        r = self.conv(x)
        r = self.reg(r)
        # residual connection
        return r + x


class SketchDenoiser(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.l0 = nn.Sequential(
            nn.Conv2d(1, d_model, 3, padding=1, dilation=1, bias=False), nn.SiLU()
        )

        self.layers = nn.Sequential(
            *(ResBlock(d_model, 3, dilation=3) for _ in range(7))
        )

        self.head = nn.Sequential(nn.Conv2d(d_model, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.l0(x)
        x = self.layers(x)
        return self.head(x)
