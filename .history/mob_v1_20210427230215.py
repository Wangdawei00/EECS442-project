import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained encoder
        self.backbone = models.mobilenet_v2(pretrained=True)

        # do not train encoder
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # get output from each layer and put into list
        layer_outputs = []
        layer_outputs.append(x)
        i = 1

        for name, module in self.backbone.features._modules.items():
            cur_input = layer_outputs[-1]
            cur_output = module(cur_input)
            layer_outputs.append(cur_output)
            i += 1

        return layer_outputs


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear', align_corners=True, slope=0.2):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode=mode, align_corners=align_corners)
        self.convA = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(slope)

    def forward(self, x, skip_connection):
        up = self.upsample(x)
        merged = torch.cat([up, skip_connection], dim=1)

        output = self.relu(self.convB(self.convA(merged)))
        return output


class Decoder(nn.Module):
    def __init__(self, encoder_out=1664, skip=256):
        super().__init__()

        # mobile
        b1_in = encoder_out + skip
        b1_out = encoder_out // 2
        b2_in = b1_out + skip // 3
        b2_out = b1_out // 2
        b3_in = b2_out + skip // 4
        b3_out = b2_out // 2
        b4_in = b3_out + skip // 6
        b4_out = b3_out // 2

        # decoder blocks
        self.conv2 = nn.Conv2d(encoder_out, encoder_out,
                               kernel_size=1, stride=1, padding=0)

        self.block1 = Decoder_block(b1_in, b1_out)
        self.block2 = Decoder_block(b2_in, b2_out)
        self.block3 = Decoder_block(b3_in, b3_out)
        self.block4 = Decoder_block(b4_in, b4_out)

        self.conv3 = nn.Conv2d(b4_out, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        conv1, pool1, pool2, pool3, encoder_output = features[
            2], features[4], features[7], features[14], features[19]

        encoder_output = self.conv2(F.relu(encoder_output))

        b1_output = self.block1(encoder_output, pool3)
        b2_output = self.block2(b1_output, pool2)
        b3_output = self.block3(b2_output, pool1)
        b4_output = self.block4(b3_output, conv1)

        depth = self.conv3(b4_output)

        return depth


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder and decoder
        self.encoder = Encoder()
        self.decoder = Decoder(1280, 96)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


