import torch
from torch import nn
from torchsummary import summary

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            x = self.relu(x)
        return x

class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.relu = nn.ReLU()

        # Conv sequence with stride support
        self.convseq = nn.Sequential(
            Conv_block(in_channels, red_channels, kernel_size=1, stride=stride, padding=0),
            Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
            Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
        )

        if in_channels == 64:
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.iden = nn.Identity()
        else:
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        y = self.convseq(x)

        x = y + self.iden(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = Conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential(
            # conv2
            self._make_layer(64, 64, 256, 3, stride=1),
            # conv3
            self._make_layer(256, 128, 512, 4, stride=2),
            # conv4
            self._make_layer(512, 256, 1024, 6, stride=2),
            # conv5
            self._make_layer(1024, 512, 2048, 3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, red_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Res_block(in_channels, red_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(Res_block(out_channels, red_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        # Weight initialization for different layers
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
    model = ResNet()
    summary(model.cuda(), (1, 224, 224))