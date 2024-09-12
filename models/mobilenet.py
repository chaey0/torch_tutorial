import torch
from torch import nn
from torchsummary import summary

#depth_wise, conv1x1, conv3x3
def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)

class InvertedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(conv_layer(in_channels, hidden_dim, kernel_size=1))  # Pointwise Conv

        layers.extend([
            conv_layer(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),  # Depthwise Conv
            conv_layer(hidden_dim, out_channels, kernel_size=1, activation=False)  # Pointwise Conv
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv_layer(in_channels, 32, kernel_size=3, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(in_channels=input_channel, out_channels=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)

        # Final Convolution and Classifier
        self.last_conv = conv_layer(input_channel, 1280, kernel_size=1, activation=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, n_classes)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

if __name__ == "__main__":
    model = MobileNetV2()
    summary(model.cuda(), (1, 224, 224))
