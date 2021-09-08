import torch.nn as nn
import torch
from core.modules import (
    ResidualConv,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
    DenseASPP,
)


class ResUnetPlusPlus(nn.Module):

    def __init__(self, n_channels,n_classes, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filters = filters

        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)



        self.denseaspp_bridge = DenseASPP(filters[2], filters[2])


        self.attn1 = AttentionBlock(filters[1], filters[2], filters[2])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[1] + filters[2], filters[1], 1, 1)



        self.attn2 = AttentionBlock(filters[0], filters[1], filters[1])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[0] + filters[1], filters[0], 1, 1)



        self.denseaspp_out = DenseASPP(filters[0], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0],n_classes, 1, 1))
        #, nn.Sigmoid()

    def forward(self, x):

        x1 = self.input_layer(x) + self.input_skip(x)

        ## 32 channels

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)
        
        ## 64 channels
        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        ## 128 channels



        x4 = self.denseaspp_bridge(x3)


        x5 = self.attn1(x2, x4)
        x5 = self.upsample1(x5)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.up_residual_conv1(x5)

        x6 = self.attn2(x1, x5)
        x6 = self.upsample2(x6)
        x6 = torch.cat([x6, x1], dim=1)
        x6 = self.up_residual_conv2(x6)


        x9 = self.denseaspp_out(x6)
        out = self.output_layer(x9)

        return out
