import torch


class SketchDenoiser(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        high_dim = d_model * 2
        low_dim = d_model

        # Stem
        self.stem = torch.nn.Conv2d(1, high_dim, 5, padding=2)
        self.bn_stem = torch.nn.BatchNorm2d(high_dim)  # Add BN here

        # Convolutions
        self.conv2 = torch.nn.Conv2d(high_dim, low_dim, 5, dilation=3, padding=6)
        self.bn2 = torch.nn.BatchNorm2d(low_dim)

        self.conv3 = torch.nn.Conv2d(low_dim, low_dim, 5, dilation=3, padding=6)
        self.bn3 = torch.nn.BatchNorm2d(low_dim)

        self.conv4 = torch.nn.Conv2d(low_dim, low_dim, 5, dilation=3, padding=6)
        self.bn4 = torch.nn.BatchNorm2d(low_dim)

        self.conv5 = torch.nn.Conv2d(low_dim, high_dim, 5, dilation=3, padding=6)
        self.bn5 = torch.nn.BatchNorm2d(high_dim)

        self.reg = torch.nn.SiLU()
        self.mlp = torch.nn.Conv2d(high_dim, 1, 1)
        self.final = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.reg(self.bn_stem(self.stem(x)))

        res = x
        x = self.reg(self.bn2(self.conv2(x)))
        x = self.reg(self.bn3(self.conv3(x)))
        x = self.reg(self.bn4(self.conv4(x)))
        x = self.reg(self.bn5(self.conv5(x)))

        return self.final(self.mlp(x + res))
