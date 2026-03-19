import torch.nn as nn
import torch
from .model_mamba2 import Mamba2
from timm.layers import DropPath
from .util import get_1d_sincos_pos_embed
import math
import numpy as np
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(1, 1))

    def forward(self, x):
        x = self.proj(x)
        return x


class CausalCNN(nn.Module):
    def __init__(self, in_channels, mid_channel, kernel_size=5):
        super(CausalCNN, self).__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn1 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn2 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn3 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size-1)
        self.bn4 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(2)

        x = self.conv1(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.transpose(1, 2)

        return x, None


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """ Local Profiling module in ARES """

    def __init__(self, in_channels, mid_channel):
        super(LocalProfiling, self).__init__()

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        )
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        )

        self.net = nn.Sequential(
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = x.squeeze(2)

        if self.training:
            sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
            x = torch.roll(x, shifts=sliding_size, dims=-1)
        else:
            sliding_size = 0

        x = self.dividing(x)
        x = self.net(x)
        x = self.combination(x)

        x = x.permute(0, 2, 1)
        return x, sliding_size


def compute_out_size(length):
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    return length


class CountMambaModel(nn.Module):
    def __init__(self, num_classes, drop_path_rate, embed_dim, depth, patch_size, max_matrix_len, early_stage, num_tabs,
                 fine_predict):
        super(CountMambaModel, self).__init__()

        self.early_stage = early_stage
        self.num_tabs = num_tabs
        self.fine_predict = fine_predict
        if fine_predict:
            assert self.num_tabs == 2

        assert num_tabs == 1 or not early_stage

        skip_localfit_flag = True

        if num_tabs == 1 or skip_localfit_flag:
            assert max_matrix_len % 6 == 0
            num_patches = max_matrix_len // 6
        else:
            assert max_matrix_len % 4 == 0
            num_patches = compute_out_size(max_matrix_len // 4) * 4
        self.num_patches = num_patches

        self.patch_embed = PatchEmbed(patch_size=(patch_size, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or skip_localfit_flag:
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
        else:
            self.local_model = LocalProfiling(in_channels=embed_dim, mid_channel=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)
        #self.blocks = nn.ModuleList([MHSA(embed_dim, embed_dim // 4) for i in range(depth)])
        self.blocks = nn.ModuleList([
            Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim // 4)
            for i in range(depth)])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)

        if fine_predict:
            self.fc = nn.Linear(embed_dim, num_classes)
            self.fc_fine = nn.Linear(embed_dim, num_classes + 1)
        else:
            self.fc = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, idx=None):
        # embed patches
        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk, drop in zip(self.blocks, self.droppaths):
            x = drop(blk(x)) + x
        x = self.fc_norm(x)
        x = x[:, 1:, :]

        if self.early_stage and self.training:
            # early-stage training
            return self.fc(x)
        if self.fine_predict:
            # fine-predict training
            x0 = self.fc_fine(x)

            x = x.mean(dim=1)
            x = self.fc(x)

            return x, x0, sliding_size

        if (self.early_stage and not self.training) or (not self.early_stage and not self.fine_predict):
            if idx is None:
                # early-stage inference or normal inference
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)

                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x
    # def forward_step(self, x, inference_params, history_feature, position_index, update):
    #     x = self.patch_embed(x).squeeze(2)
    #
    #     # local model
    #     conv_state_1, conv_state_2, conv_state_3, conv_state_4 = self._get_conv_states_from_cache(inference_params)
    #
    #     if conv_state_1.dim() == 3:
    #         conv_state_1 = conv_state_1.unsqueeze(-1)  # 变成 [1, 256, 4, 1]
    #     print(x.shape, conv_state_1.shape)
    #     # Conv1
    #     x = torch.cat([conv_state_1, x], dim=-1)
    #     if update:
    #         conv_state_1.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
    #     x = self.local_model.conv1(x)
    #     x = self.local_model.bn1(x)
    #     x = self.local_model.relu1(x)
    #
    #     # Conv2
    #     x = torch.cat([conv_state_2, x], dim=-1)
    #     if update:
    #         conv_state_2.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
    #     x = self.local_model.conv2(x)
    #     x = self.local_model.bn2(x)
    #     x = self.local_model.relu2(x)
    #
    #     # Pool1
    #     x = self.local_model.pool1(x)
    #     x = self.local_model.dropout1(x)
    #
    #     # Conv3
    #     x = torch.cat([conv_state_3, x], dim=-1)
    #     if update:
    #         conv_state_3.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
    #     x = self.local_model.conv3(x)
    #     x = self.local_model.bn3(x)
    #     x = self.local_model.relu3(x)
    #
    #     # Conv4
    #     x = torch.cat([conv_state_4, x], dim=-1)
    #     if update:
    #         conv_state_4.copy_(x[:, :, -(self.local_model.kernel_size - 1):])
    #     x = self.local_model.conv4(x)
    #     x = self.local_model.bn4(x)
    #     x = self.local_model.relu4(x)
    #
    #     # Pool2
    #     x = self.local_model.pool2(x)
    #     x = self.local_model.dropout2(x)
    #
    #     # pos embedding
    #     x = x.transpose(1, 2)
    #
    #     clip_position_index = min(position_index, 299)
    #     x = x + self.pos_embed[:, 1 + clip_position_index, :]
    #     if clip_position_index == 0:
    #         cls_x = self.cls_token + self.pos_embed[:, :1, :]
    #         cls_x = cls_x.squeeze(1)
    #
    #         for blk, drop in zip(self.blocks, self.droppaths):
    #             cls_x = drop(blk.forward_stage(cls_x, inference_params, update)) + cls_x
    #
    #     # apply Transformer blocks
    #     for blk, drop in zip(self.blocks, self.droppaths):
    #         x = drop(blk.forward_stage(x, inference_params, update)) + x
    #
    #     x = self.fc_norm(x)
    #
    #     history_feature = (history_feature * position_index + x) / (position_index + 1)
    #     x = self.fc(history_feature)
    #
    #     return x, history_feature
    # def _get_conv_states_from_cache(self, inference_params):
    #     if "conv" not in inference_params.key_value_memory_dict:
    #         conv_state_1 = torch.zeros(
    #             1,
    #             self.local_model.conv1.weight.shape[0],
    #             self.local_model.kernel_size - 1,
    #             device=self.local_model.conv1.weight.device,
    #             dtype=self.local_model.conv1.weight.dtype,
    #         )
    #         conv_state_2 = torch.zeros(
    #             1,
    #             self.local_model.conv2.weight.shape[0],
    #             self.local_model.kernel_size - 1,
    #             device=self.local_model.conv2.weight.device,
    #             dtype=self.local_model.conv2.weight.dtype,
    #         )
    #         conv_state_3 = torch.zeros(
    #             1,
    #             self.local_model.conv3.weight.shape[0],
    #             self.local_model.kernel_size - 1,
    #             device=self.local_model.conv3.weight.device,
    #             dtype=self.local_model.conv3.weight.dtype,
    #         )
    #         conv_state_4 = torch.zeros(
    #             1,
    #             self.local_model.conv4.weight.shape[0],
    #             self.local_model.kernel_size - 1,
    #             device=self.local_model.conv4.weight.device,
    #             dtype=self.local_model.conv4.weight.dtype,
    #         )
    #
    #         inference_params.key_value_memory_dict["conv"] = (conv_state_1, conv_state_2, conv_state_3, conv_state_4)
    #
    #     conv_state = inference_params.key_value_memory_dict["conv"]
    #     return conv_state
class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        return self.mhsa(x, x, x)[0]

class CountMambaModel_old(nn.Module):
    def __init__(self, num_classes, drop_path_rate, embed_dim, depth, patch_size, max_matrix_len, early_stage, num_tabs,
                 fine_predict):
        super().__init__()

        self.early_stage = early_stage
        self.num_tabs = num_tabs
        self.fine_predict = fine_predict
        if fine_predict:
            assert self.num_tabs == 2

        assert num_tabs == 1 or not early_stage

        skip_localfit_flag = False

        if num_tabs == 1 or skip_localfit_flag:
            assert max_matrix_len % 6 == 0
            num_patches = max_matrix_len // 6
        else:
            assert max_matrix_len % 4 == 0
            num_patches = compute_out_size(max_matrix_len // 4) * 4
        self.num_patches = num_patches

        self.patch_embed = PatchEmbed(patch_size=(patch_size, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or skip_localfit_flag:
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
        else:
            self.local_model = LocalProfiling(in_channels=embed_dim, mid_channel=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim//4)
            for i in range(depth)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)

        if fine_predict:
            self.fc = nn.Linear(embed_dim, num_classes)
            self.fc_fine = nn.Linear(embed_dim, num_classes + 1)
        else:
            self.fc = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, idx=None):
        # embed patches
        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk, drop in zip(self.blocks, self.droppaths):
            x = drop(blk(x)) + x
        x = self.fc_norm(x)
        x = x[:, 1:, :]

        if self.early_stage and self.training:
            # early-stage training
            return self.fc(x)
        if self.fine_predict:
            # fine-predict training
            x0 = self.fc_fine(x)

            x = x.mean(dim=1)
            x = self.fc(x)

            return x, x0, sliding_size

        if (self.early_stage and not self.training) or (not self.early_stage and not self.fine_predict):
            if idx is None:
                # early-stage inference or normal inference
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)

                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x
if __name__ == "__main__":
    model = CountMambaModel(num_classes=10, drop_path_rate=0.1, embed_dim=192, depth=12, patch_size=8,
                            max_matrix_len=7200, early_stage=False, num_tabs=2, fine_predict=False)
    from torchinfo import summary
    summary(model, input_size=(10, 1, 8, 7200), col_names=["input_size", "output_size", "num_params"])
