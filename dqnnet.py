import math
import torch.nn as nn


class DQNNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQNNet, self).__init__()
        self.stem = nn.Sequential(*[
            self.make_conv_block(in_channels, 32, kernel_size=7, stride=4),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.make_conv_block(32, 64, kernel_size=5, stride=2),
            self.make_conv_block(64, 64, kernel_size=3, stride=1),
        ])
        self.head = nn.Sequential(*[
            nn.Linear(1600, 512),
            nn.Linear(512, num_actions),
        ])
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        return self.head(x)

    @staticmethod
    def make_conv_block(in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=None):
        padding = kernel_size // 2 if padding is None else padding
        return nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
        ])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
