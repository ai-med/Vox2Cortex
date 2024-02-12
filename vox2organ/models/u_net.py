""" UNet architecture """

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.custom_layers import IdLayer

class ResidualBlock(nn.Module):
    """ Residual Block of https://arxiv.org/abs/1908.02182,
    implementation at https://github.com/MIC-DKFZ/nnUNet
    """

    def __init__(self, num_channels_in: int, num_channels_out: int,
                 normalize='batch', p_dropout: float=None, ndims=3):

        super().__init__()
        if ndims == 3:
            ConvLayer = nn.Conv3d
            if normalize == 'batch':
                norm = nn.BatchNorm3d
            elif normalize == 'layer':
                norm = nn.LayerNorm
            elif normalize == 'instance':
                norm = nn.InstanceNorm3d
            else:
                raise ValueError("Invalid normalization.")
        elif ndims == 2:
            ConvLayer = nn.Conv2d
            if normalize == 'batch':
                norm = nn.BatchNorm2d
            elif normalize == 'layer':
                norm = nn.LayerNorm
            elif normalize == 'instance':
                norm = nn.InstanceNorm2d
            else:
                raise ValueError("Invalid normalization.")
        else:
            raise ValueError("Invalid number of dimensions.")
        #TODO: Bias can probably be removed (canceled out by norm anyways)
        self.conv1 = ConvLayer(num_channels_in, num_channels_out,
                               kernel_size=3, padding=1)
        self.norm1 = norm(num_channels_out)
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = IdLayer()

        #TODO: Bias can probably be removed (canceled out by norm anyways)
        self.conv2 = ConvLayer(
            num_channels_out, num_channels_out, kernel_size=3, padding=1
        )
        self.norm2 = norm(num_channels_out)

        # 1x1x1 conv to adapt channels of residual
        if num_channels_in != num_channels_out:
            self.adapt_skip = nn.Sequential(ConvLayer(num_channels_in,
                                                      num_channels_out, 1,
                                                      bias=False),
                                            norm(num_channels_out))
        else:
            self.adapt_skip = IdLayer()

    def forward(self, x):
        # Conv --> Norm --> ReLU --> (Dropout)
        x_out = self.conv1(x)
        x_out = F.relu(self.norm1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.norm2(self.conv2(x_out))
        res = self.adapt_skip(x)
        x_out += res

        return F.relu(x_out)

class ResidualUNetEncoder(nn.Module):
    """ Residual UNet encoder oriented on https://github.com/MIC-DKFZ/nnUNet.

    :param input_channels: The number of image channels
    :param encoder_channels: List of channel dimensions of all feature maps
    :returns: Encoded feature maps for every encoder step
    """
    def __init__(
        self,
        input_channels: int,
        encoder_channels,
        p_dropout: float,
        ndims=3,
        norm='batch',
    ):
        super().__init__()

        self.num_steps = len(encoder_channels)
        self.channels = encoder_channels

        if ndims == 3:
            ConvLayer = nn.Conv3d
        elif ndims == 2:
            ConvLayer = nn.Conv2d
        else:
            raise ValueError("Invalid number of dimensions.")

        # Initial step: Conv --> Residual block
        down_layers = [nn.Sequential(
            ConvLayer(input_channels, self.channels[0], 3, padding=1),
            ResidualBlock(
                self.channels[0],
                self.channels[0],
                p_dropout=p_dropout,
                ndims=ndims,
                normalize=norm,
            )
        )]
        for i in range(1, self.num_steps):
            # Downsample --> Residual Block
            down_layers.append(nn.Sequential(
                # Compared to Kong et al. we use 2x2x2 convs (instead of 3x3x3)
                # for downsampling
                ConvLayer(self.channels[i-1], self.channels[i], 2, stride=2),
                ResidualBlock(
                    self.channels[i],
                    self.channels[i],
                    p_dropout=p_dropout,
                    ndims=ndims,
                    normalize=norm
                )
            ))

        self.encoder = nn.ModuleList(down_layers)

    def forward(self, x):
        skips = []

        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        return skips

class ResidualUNetDecoder(nn.Module):
    """ Residual UNet decoder oriented on https://github.com/MIC-DKFZ/nnUNet.

    :param encoder: The encoder from which the decoder receives features
    :param decoder_channels: List of channel dimensions of all feature maps
    :param num_classes: The number of classes to segment
    :returns: Segmentation output
    """
    def __init__(
        self,
        encoder,
        decoder_channels,
        num_classes,
        patch_shape,
        deep_supervision,
        p_dropout,
        ndims=3,
        init_last_zero=False,
        norm='batch',
    ):
        super().__init__()
        # Decoder has one step less
        num_steps = encoder.num_steps - 1
        self.num_classes = num_classes
        self.channels = decoder_channels
        self.deep_supervision = deep_supervision
        self.deep_supervision_pos = (1, 2)
        deep_supervision_layers = []

        if ndims == 3:
            ConvLayer = nn.Conv3d
            ConvTransposeLayer = nn.ConvTranspose3d
        elif ndims == 2:
            ConvLayer = nn.Conv2d
            ConvTransposeLayer = nn.ConvTranspose2d
        else:
            raise ValueError("Invalid number of dimensions.")

        up_layers = []
        # First decoder step: Upsample --> Residual Blocks
        up_layers.append(nn.Sequential(
            ConvTransposeLayer(
                encoder.channels[-1],
                self.channels[0],
                kernel_size=2,
                stride=2
            ),
            ResidualBlock(
                self.channels[0], self.channels[0],
                p_dropout=p_dropout,
                ndims=ndims,
                normalize=norm,
            )
        ))

        # Decoder steps: Upsample --> Residual Blocks
        for i in range(1, num_steps):
            up_layers.append(nn.Sequential(
                ConvTransposeLayer(
                    self.channels[i-1],
                    self.channels[i],
                    kernel_size=2,
                    stride=2
                ),
                ResidualBlock(
                    self.channels[i],
                    self.channels[i],
                    p_dropout=p_dropout,
                    ndims=ndims,
                    normalize=norm,
                )
            ))
            if deep_supervision and i in self.deep_supervision_pos:
                mode = 'trilinear' if ndims == 3 else 'bilinear'
                deep_supervision_layers.append(nn.Sequential(
                    nn.Upsample(patch_shape, mode=mode,
                                align_corners=False),
                    ConvLayer(self.channels[i], num_classes, 1, bias=False)
                ))

        if deep_supervision:
            self.deep_supervision_layers = nn.ModuleList(
                deep_supervision_layers
            )

        # Segmenation layer
        self.final_layer = ConvLayer(
            self.channels[-1], num_classes, 1, bias=False
        )

        # Initialize the last layer with zeros
        if init_last_zero:
            nn.init.constant_(self.final_layer.weight, 0.0)

        self.decoder = nn.ModuleList(up_layers)

    def forward(self, skips):
        # Reverse order of skips from encoder
        down_skips = skips[::-1]

        x = down_skips[0]

        up_skips = []
        seg = []
        i_ds = 0 # For counting deep supervisions

        for i, layer in enumerate(self.decoder):
            # Upsample
            x = layer[0](x)
            x = torch.sum(torch.stack([x, down_skips[i+1]]), dim=0)
            # Residual block
            x = layer[1](x)
            up_skips.append(x)

            if self.deep_supervision and i in self.deep_supervision_pos:
                seg.append(self.deep_supervision_layers[i_ds](x))
                i_ds += 1

        # Final output belongs to the segmentations and to the skips
        final_out = self.final_layer(x)
        up_skips.append(final_out)
        seg.append(final_out)

        return up_skips, seg


class ResidualUNet(nn.Module):
    """ Residual UNet oriented on https://github.com/MIC-DKFZ/nnUNet.
    It allows to flexibly exchange the size of the decoder and to get feature
    maps from different stages of the encoder and/or decoder.
    """
    def __init__(
        self,
        num_classes: int,
        num_input_channels: int,
        patch_shape,
        down_channels,
        up_channels,
        deep_supervision,
        voxel_decoder: bool,
        p_dropout: float=None,
        ndims=3,
        init_last_zero=False,
        norm='batch',
    ):
        assert len(up_channels) == len(down_channels) - 1,\
                "Encoder should have one more step than decoder."
        super().__init__()
        self.num_classes = num_classes

        self.encoder = ResidualUNetEncoder(
            num_input_channels,
            down_channels,
            p_dropout,
            ndims=ndims,
            norm=norm,
        )
        if voxel_decoder:
            self.decoder = ResidualUNetDecoder(
                self.encoder,
                up_channels,
                num_classes,
                patch_shape,
                deep_supervision,
                p_dropout,
                ndims=ndims,
                init_last_zero=init_last_zero,
                norm=norm,
            )
        else:
            self.decoder = None

    def forward(self, x):
        # Input image
        encoder_skips = [x]
        # Features from encoder
        encoder_skips += self.encoder(x)
        # Features from decoder
        if self.decoder is not None:
            decoder_skips, seg_out = self.decoder(encoder_skips)
        else:
            decoder_skips, seg_out = [], []

        return encoder_skips, decoder_skips, seg_out
