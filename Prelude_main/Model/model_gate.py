import torch.nn as nn
import math
import torch


class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        self.feature_dim = 512
        # RF_model 现在会返回确定的特征维度
        self.model = RF_model(out_feature_dim=self.feature_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.gating_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.model(x)
        x = self.gap(feat)
        x = x.view(x.size(0), -1)
        h = self.gating_fc(x)
        return h


class RF_model(nn.Module):
    def __init__(self, out_feature_dim=512):
        super(RF_model, self).__init__()
        self.first_layer = make_first_layers()

        # 核心修正：make_first_layers 最终输出的是 64 个通道
        # 如果经过 view 转换，我们需要确保 1D 卷积的输入通道匹配
        self.features = make_layers([128, 128, 'M', 256, 256, 'M', 512, out_feature_dim], in_channels=64)

        self._initialize_weights()

    def forward(self, x):
        # 1. 经过 2D 卷积层: input [B, 1, 5, 1800] -> output [B, 64, H', W']
        x = self.first_layer(x)

        # 2. 核心修正：处理维度以适配 1D 卷积
        # 这里的 H' 因为卷积和池化会变小。报错显示你的 H' 此时被 view 压进了通道。
        # 我们使用 mean 或者是 view 重新排列，确保通道数依然是 64
        # 假设我们只关心时间维度上的特征映射：
        x = torch.mean(x, dim=2)  # 将高度维度 H' 压缩，保持通道数为 64, 形状变为 [B, 64, W']

        # 3. 经过 1D 卷积层
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=64):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3, stride=2, padding=1), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d((1, 3)),
        nn.Dropout(0.1),
        nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),  # 经过这一步，Height 维度通常会变小
        nn.Dropout(0.1)
    )


if __name__ == "__main__":
    from torchinfo import summary

    B, C, T = 20, 5, 1800
    model = GateNet()
    # 打印模型结构验证
    summary(model, input_size=(B, 1, C, T))