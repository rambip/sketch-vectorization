import torch


class Rasterizer(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, d_model, 11, padding=5)
        self.reg1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(d_model, d_model, 5, dilation=3, padding=6)
        self.reg2 = torch.nn.ReLU()

        # self.conv3 = torch.nn.Conv2d(d_model, 1, 5, dilation=10, padding=20)
        self.conv3 = torch.nn.Conv2d(d_model, 1, 1)
        self.reg3 = torch.nn.Sigmoid()

        # self.mlp = torch.nn.Conv2d(d_model, 1, 1)
        # self.final = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.reg1(self.conv1(x))
        x = self.reg2(self.conv2(x))
        x = self.reg3(self.conv3(x))
        return x
