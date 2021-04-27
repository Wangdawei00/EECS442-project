import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained encoder
        self.backbone = models.densenet169(pretrained=True)

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

        b1_in = encoder_out + skip
        b1_out = encoder_out // 2
        b2_in = b1_out + skip // 2
        b2_out = b1_out // 2
        b3_in = b2_out + skip // 4
        b3_out = b2_out // 2
        b4_in = b3_out + skip // 4
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
            3], features[4], features[6], features[8], features[12]

        encoder_output = self.conv2(F.relu(encoder_output))

        b1_output = self.block1(encoder_output, pool3)
        b2_output = self.block2(b1_output, pool2)
        b3_output = self.block3(b2_output, pool1)
        b4_output = self.block4(b3_output, conv1)

        depth = self.conv3(b4_output)

        return depth


class Dense169(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder and decoder
        self.encoder = Encoder()
        # TODO
        self.decoder = Decoder(1664, 256)

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


"""
dense169
3 4 6 8 12
1664 256
feature1: conv0         torch.Size([5, 64, 240, 320])
feature2: norm0         torch.Size([5, 64, 240, 320])
feature3: relu0         torch.Size([5, 64, 240, 320])
feature4: pool0         torch.Size([5, 64, 120, 160])
feature5: denseblock1           torch.Size([5, 256, 120, 160])
feature6: transition1           torch.Size([5, 128, 60, 80])
feature7: denseblock2           torch.Size([5, 512, 60, 80])
feature8: transition2           torch.Size([5, 256, 30, 40])
feature9: denseblock3           torch.Size([5, 1280, 30, 40])
feature10: transition3          torch.Size([5, 640, 15, 20])
feature11: denseblock4          torch.Size([5, 1664, 15, 20])
feature12: norm5                torch.Size([5, 1664, 15, 20])
torch.Size([5, 1920, 30, 40])
torch.Size([5, 960, 60, 80])
torch.Size([5, 480, 120, 160])
torch.Size([5, 272, 240, 320])
torch.Size([5, 1, 240, 320])
"""

"""
dense121
3 4 6 8 12
1024 256
feature1: conv0         torch.Size([5, 64, 240, 320])
feature2: norm0         torch.Size([5, 64, 240, 320])
feature3: relu0         torch.Size([5, 64, 240, 320])
feature4: pool0         torch.Size([5, 64, 120, 160])
feature5: denseblock1           torch.Size([5, 256, 120, 160])
feature6: transition1           torch.Size([5, 128, 60, 80])
feature7: denseblock2           torch.Size([5, 512, 60, 80])
feature8: transition2           torch.Size([5, 256, 30, 40])
feature9: denseblock3           torch.Size([5, 1024, 30, 40])
feature10: transition3          torch.Size([5, 512, 15, 20])
feature11: denseblock4          torch.Size([5, 1024, 15, 20])
feature12: norm5                torch.Size([5, 1024, 15, 20])
torch.Size([5, 1280, 30, 40])
torch.Size([5, 640, 60, 80])
torch.Size([5, 320, 120, 160])
torch.Size([5, 192, 240, 320])
torch.Size([5, 1, 240, 320])
"""


"""
res50
3 4 6 7 8
2048 1024
feature1: conv1         torch.Size([5, 64, 240, 320])
feature2: bn1           torch.Size([5, 64, 240, 320])
feature3: relu          torch.Size([5, 64, 240, 320])
feature4: maxpool               torch.Size([5, 64, 120, 160])
feature5: layer1                torch.Size([5, 256, 120, 160])
feature6: layer2                torch.Size([5, 512, 60, 80])
feature7: layer3                torch.Size([5, 1024, 30, 40])
feature8: layer4                torch.Size([5, 2048, 15, 20])
torch.Size([5, 3072, 30, 40])
torch.Size([5, 1536, 60, 80])
torch.Size([5, 576, 120, 160])
torch.Size([5, 320, 240, 320])
torch.Size([5, 1, 240, 320])
"""

"""
mobilenet_v2
2 4 7 14 19
1280 96
feature1: 0             torch.Size([5, 32, 240, 320])
feature2: 1             torch.Size([5, 16, 240, 320])
feature3: 2             torch.Size([5, 24, 120, 160])
feature4: 3             torch.Size([5, 24, 120, 160])
feature5: 4             torch.Size([5, 32, 60, 80])
feature6: 5             torch.Size([5, 32, 60, 80])
feature7: 6             torch.Size([5, 32, 60, 80])
feature8: 7             torch.Size([5, 64, 30, 40])
feature9: 8             torch.Size([5, 64, 30, 40])
feature10: 9            torch.Size([5, 64, 30, 40])
feature11: 10           torch.Size([5, 64, 30, 40])
feature12: 11           torch.Size([5, 96, 30, 40])
feature13: 12           torch.Size([5, 96, 30, 40])
feature14: 13           torch.Size([5, 96, 30, 40])
feature15: 14           torch.Size([5, 160, 15, 20])
feature16: 15           torch.Size([5, 160, 15, 20])
feature17: 16           torch.Size([5, 160, 15, 20])
feature18: 17           torch.Size([5, 320, 15, 20])
feature19: 18           torch.Size([5, 1280, 15, 20])
torch.Size([5, 1376, 30, 40])
torch.Size([5, 672, 60, 80])
torch.Size([5, 344, 120, 160])
torch.Size([5, 176, 240, 320])
torch.Size([5, 1, 240, 320])
"""
