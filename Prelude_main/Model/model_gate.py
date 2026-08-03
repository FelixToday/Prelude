import torch.nn as nn
import math
import torch


# ============================================================
# 三阶段重构：UnifiedGateNet
# ============================================================

class UnifiedGateNet(nn.Module):
    """
    统一的轻量门控网络，接收原始流量数据 (B, 2, L)，通过 1D 卷积提取特征后输出置信度。
    
    输入: (B, 2, L) — 原始流量，通道 0 为时间戳，通道 1 为包大小/方向
    输出: (B, 1) — [0, 1] 置信度（Sigmoid）
    
    设计要点：
    - 与 backbone 的输入格式完全解耦
    - 轻量级架构，仅用 1D 卷积
    - 截断后的流量中超出截断点的数据包已置零，网络自动学习识别有效信息量
    """
    def __init__(self, seq_len=5000, hidden_dim=64):
        super(UnifiedGateNet, self).__init__()
        self.seq_len = seq_len
        
        # 1D 卷积特征提取器
        self.conv_net = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.1),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # 门控分类头
        self.gating_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, 2, L) 原始流量，通道 0=时间戳，通道 1=包大小
        Returns:
            h: (B, 1) [0, 1] 置信度
        """
        feat = self.conv_net(x)
        h = feat.view(feat.size(0), -1)
        h = self.gating_head(h)
        return h