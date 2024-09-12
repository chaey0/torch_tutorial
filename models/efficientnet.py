import torch
from torch import nn
import random
from torchsummary import summary
#EfficientNet
basic_mb_params = [
    # (expand_ratio, out_channels, repeats, stride, kernel_size)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)

class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, expand_ratio, reduction=2, survival_prob=0.8):
        super(MBBlock, self).__init__()

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        # Squeeze-and-Excitation에서 사용할 축소된 차원
        reduced_dim = int(in_channels / reduction)

        # 확장 단계
        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)

        # Depthwise Convolution
        self.depthwise_conv = ConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=hidden_dim)

        # Squeeze-and-Excitation 블록
        self.se_block = SqueezeExcitation(hidden_dim, reduced_dim)

        # Pointwise Convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.use_residual = in_channels == out_channels and stride == 1
        self.survival_prob = survival_prob if self.use_residual else 1.0

    def forward(self, inputs):
        # Stochastic Depth
        if self.training and random.random() > self.survival_prob:
            return inputs

        if self.expand:
            x = self.expand_conv(inputs)
        else:
            x = inputs

        x = self.depthwise_conv(x)
        x = self.se_block(x)
        x = self.pointwise_conv(x)

        if self.use_residual:
            x = x + inputs

        return x

class EfficientNet(nn.Module):
    def __init__(self, version="b0", num_classes=10):
        super(EfficientNet, self).__init__()

        phi, resolution, dropout_rate = scale_values[version]
        self.phi = phi
        self.resolution = resolution
        self.dropout_rate = dropout_rate

        self.scale_depth = lambda t: int(t * alpha ** phi)
        self.scale_width = lambda c: int(c * beta ** phi)

        self.stem_conv = ConvBlock(1, self.scale_width(32), kernel_size=3, stride=2, padding=1)

        # MBConv blocks
        self.layers = self.create_mbconv_layers(basic_mb_params)

        self.head_conv = ConvBlock(self.scale_width(320), self.scale_width(1280), kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(self.scale_width(1280), num_classes)

        # Initialize weights
        self._initialize_weights()

    def create_mbconv_layers(self, mb_params):
        layers = []
        input_channels = self.scale_width(32)

        # MBConv 블록을 반복하며 생성
        for expand_ratio, out_channels, repeats, stride, kernel_size in mb_params:
            output_channels = self.scale_width(out_channels)
            for i in range(self.scale_depth(repeats)):
                stride_ = stride if i == 0 else 1  # 첫 번째 반복에만 stride 적용
                layers.append(
                    MBBlock(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        expand_ratio=expand_ratio,
                        stride=stride_,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
                input_channels = output_channels  # 출력 채널을 다음 블록의 입력 채널로 설정
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the FC layer
        x = self.dropout(x)
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
    model = EfficientNet()
    summary(model.cuda(), (1, 224, 224))