import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    constant_init,
    kaiming_init,
    normal_init,
)
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.models.utils.ops import resize
from mmpose.utils import get_root_logger
from mmpose.models.builder import BACKBONES


class KernelAttention(nn.Module):
    def __init__(
            self,
            channels,
            reduction=4,
            num_kernels=4,
            init_weight=True,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN")
    ):
        super().__init__()
        if channels != 3:
            mid_channels = channels // reduction
        else:
            mid_channels = num_kernels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = build_conv_layer(
            conv_cfg,
            channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm = build_norm_layer(norm_cfg, mid_channels)[1]
        self.act = act_layer()

        self.conv2 = build_conv_layer(
            conv_cfg,
            mid_channels,
            num_kernels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)

        x = self.conv2(x).view(x.shape[0], -1)
        x = self.sigmoid(x)

        return x


class KernelAggregation(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            num_kernels,
            init_weight=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_kernels = num_kernels

        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(num_kernels, out_channels)
            )
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        batch_size, in_channels, height, width = x.size()

        x = x.contiguous().view(1, batch_size * self.in_channels, height, width)

        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(
                x,
                weight=weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )
        else:
            x = F.conv2d(
                x,
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )

        x = x.contiguous().view(batch_size, self.out_channels, x.shape[-2], x.shape[-1])

        return x


class DynamicKernelAggregation(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            num_kernels=4,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN")
    ):
        super().__init__()
        assert in_channels % groups == 0

        self.attention = KernelAttention(
            in_channels,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.aggregation = KernelAggregation(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels
        )

    def forward(self, x):
        attention = x

        attention = self.attention(attention)
        x = self.aggregation(x, attention)

        return x


class DConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            num_kernels=1,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN")
    ):
        super().__init__()
        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                num_kernels=num_kernels,
                act_layer=act_layer,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
        else:
            self.conv = build_conv_layer(
                conv_cfg,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class DynamicLocalPerception(nn.Module):
    def __init__(
            self,
            dim,
            num_kernels=1,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            with_cp=False
    ):
        super().__init__()
        self.with_cp = with_cp

        self.dw3x3 = DConv(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=False,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.act = act_layer()

    def forward(self, x):

        def _inner_forward(x):
            x = self.act(self.norm(self.dw3x3(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class GlobalSpatialAttention(nn.Module):
    def __init__(
            self,
            dim,
            conv_cfg=None,
            norm_cfg=dict(type="BN")
    ):
        super().__init__()
        # depth-wise convolution
        self.dw = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=dim,
            bias=False
        )
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]

        # depth-wise dilation convolution
        self.dwd = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3,
            groups=dim,
            bias=False
        )
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

        # channel convolution
        self.pw = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm3 = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x):
        attn = x.clone()

        attn = self.norm1(self.dw(attn))
        attn = self.norm2(self.dwd(attn))
        attn = self.norm3(self.pw(attn))

        return x * attn


class GlobalFFN(nn.Module):
    def __init__(
            self,
            dim,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            with_cp=False
    ):
        super().__init__()
        self.with_cp = with_cp

        self.proj1 = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.act = act_layer()

        self.spatial_gate = GlobalSpatialAttention(
            dim,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.proj2 = build_conv_layer(
            conv_cfg,
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x.clone()

            x = self.proj1(x)
            x = self.norm1(x)
            x = self.act(x)

            x = self.spatial_gate(x)

            x = self.proj2(x)
            x = self.norm2(x)

            x = x + shortcut

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class DynamicFFN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            num_kernels=1,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            with_cp=False
    ):
        super().__init__()
        self.with_cp = with_cp

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.proj1 = DConv(
            in_features,
            hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]

        self.dw = DConv(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=False,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.norm2 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.act = act_layer()

        self.proj2 = DConv(
            hidden_features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )
        self.norm3 = build_norm_layer(norm_cfg, out_features)[1]

    def forward(self, x):

        def _inner_forward(x):
            x = self.proj1(x)
            x = self.norm1(x)

            x = self.dw(x)
            x = self.norm2(x)
            x = self.act(x)

            x = self.proj2(x)
            x = self.norm3(x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class HighResolutionContextBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            dim,
            mlp_ratio=4,
            num_kernels=1,
            drop_path=0.0,
            act_layer=nn.GELU,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            with_cp=False
    ):
        super(HighResolutionContextBlock, self).__init__()
        self.with_cp = with_cp

        self.dlp = DynamicLocalPerception(
            dim,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp1 = GlobalFFN(
            dim,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        hidden_dim = int(dim * mlp_ratio)
        self.mlp2 = DynamicFFN(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            num_kernels=num_kernels,
            act_layer=act_layer,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        )

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path)) if drop_path > 0.0 else nn.Identity()

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):

        def _inner_forward(x):
            # DynamicLocalPerception
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dlp(x))

            # GlobalFFN
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp1(self.norm1(x)))

            # DynamicFFN
            x = x + self.drop_path(self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * self.mlp2(self.norm2(x)))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class LayerNormWithReshaping(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class HighResolutionContextModule(nn.Module):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            in_channels,
            num_channels,
            multiscale_output,
            with_cp=False,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            num_mlp_ratios=None,
            num_kernels=None,
            drop_paths=0.0
    ):
        super(HighResolutionContextModule, self).__init__()
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_mlp_ratios,
            num_kernels,
            drop_paths
        )
        self.fuse_layers = self._make_fuse_layers()
        self.act = nn.GELU()

    def _make_one_branch(
            self,
            branch_index,
            block,
            num_blocks,
            num_channels,
            num_mlp_ratios,
            num_kernels,
            drop_paths
    ):
        """Make one branch."""
        layers = []

        layers.append(
            block(
                self.in_channels[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                num_kernels=num_kernels[branch_index],
                drop_path=drop_paths[0],
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                with_cp=self.with_cp
            )
        )
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    num_kernels=num_kernels[branch_index],
                    drop_path=drop_paths[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp
                )
            )

        layers.append(LayerNormWithReshaping(self.in_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(
            self,
            num_branches,
            block,
            num_blocks,
            num_channels,
            num_mlp_ratios,
            num_kernels,
            drop_paths
    ):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_mlp_ratios,
                    num_kernels,
                    drop_paths
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        num_out_branches = num_branches if self.multiscale_output else 1

        fuse_layers = []
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode="bilinear",
                                align_corners=False
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.GELU(),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + resize(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.act(y))

        return x_fuse


@BACKBONES.register_module()
class HRNeXt(nn.Module):
    """HRNeXt backbone.
    High Resolution Context Network Backbone
    """

    blocks_dict = {
        "HRNEXT_BLOCK": HighResolutionContextBlock
    }

    def __init__(
            self,
            extra,
            in_channels=3,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            norm_eval=False,
            with_cp=False
    ):
        super(HRNeXt, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        # generate drop path rate list
        depth_s1 = (
                self.extra["stage1"]["num_blocks"][0] * self.extra["stage1"]["num_modules"]
        )
        depth_s2 = (
                self.extra["stage2"]["num_blocks"][0] * self.extra["stage2"]["num_modules"]
        )
        depth_s3 = (
                self.extra["stage3"]["num_blocks"][0] * self.extra["stage3"]["num_modules"]
        )
        depth_s4 = (
                self.extra["stage4"]["num_blocks"][0] * self.extra["stage4"]["num_modules"]
        )
        depths = [depth_s1, depth_s2, depth_s3, depth_s4]
        drop_path_rate = self.extra["drop_path_rate"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        logger = get_root_logger()
        logger.info(dpr)

        # stage 1
        self.stage1_cfg = self.extra["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block_type = self.stage1_cfg["block"]
        num_blocks = self.stage1_cfg["num_blocks"][0]
        num_mlp_ratios = self.stage1_cfg["num_mlp_ratios"][0]
        num_kernels = self.stage1_cfg["num_kernels"][0]

        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, num_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, num_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, num_channels, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(self.norm_cfg, num_channels, postfix=4)
        self.act = nn.GELU()

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=num_channels,
            bias=False
        )
        self.add_module(self.norm2_name, norm2)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            num_channels,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.conv4 = build_conv_layer(
            self.conv_cfg,
            num_channels,
            num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=num_channels,
            bias=False
        )
        self.add_module(self.norm4_name, norm4)

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(
            block,
            num_channels,
            num_blocks,
            num_mlp_ratios,
            num_kernels,
            drop_paths=dpr[0:depth_s1]
        )

        # stage 2
        self.stage2_cfg = self.extra["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block_type = self.stage2_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels],
            num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg,
            num_channels,
            drop_paths=dpr[depth_s1:depth_s1 + depth_s2]
        )

        # stage 3
        self.stage3_cfg = self.extra["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block_type = self.stage3_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s1 + depth_s2: depth_s1 + depth_s2 + depth_s3]
        )

        # stage 4
        self.stage4_cfg = self.extra["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block_type = self.stage4_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            drop_paths=dpr[depth_s1 + depth_s2 + depth_s3:],
            multiscale_output=False
        )

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm4_name)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1],
                            nn.GELU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.GELU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(
            self,
            block,
            num_channels,
            blocks,
            mlp_ratio,
            num_kernels,
            drop_paths
    ):
        """Make each layer."""
        layers = []

        for i in range(0, blocks):
            layers.append(
                block(
                    num_channels,
                    mlp_ratio=mlp_ratio,
                    num_kernels=num_kernels,
                    drop_path=drop_paths[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp
                )
            )

        layers.append(LayerNormWithReshaping(num_channels))

        return nn.Sequential(*layers)

    def _make_stage(
            self,
            layer_config,
            in_channels,
            drop_paths,
            multiscale_output=True
    ):
        """Make each stage."""
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        block = self.blocks_dict[layer_config["block"]]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        num_mlp_ratios = layer_config["num_mlp_ratios"]
        num_kernels = layer_config["num_kernels"]

        hra_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hra_modules.append(
                HighResolutionContextModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    num_mlp_ratios=num_mlp_ratios,
                    num_kernels=num_kernels,
                    drop_paths=drop_paths[num_blocks[0] * i: num_blocks[0] * (i + 1)]
                )
            )

        return nn.Sequential(*hra_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pretrained weights.
            Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            ckpt = load_checkpoint(self, pretrained, strict=False)
            if "model" in ckpt:
                msg = self.load_state_dict(ckpt["model"], strict=False)
                logger.info(msg)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        """Forward function."""
        # Stem
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act(x)

        # Stage1
        x = self.layer1(x)

        # Stage2
        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage3
        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage4
        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super(HRNeXt, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
