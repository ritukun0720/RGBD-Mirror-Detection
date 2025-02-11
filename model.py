import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pdb
import numpy as np
from torchvision.models import resnext101_32x8d

class ResNeXtEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNeXtEncoder, self).__init__()
        resnext = resnext101_32x8d(pretrained=pretrained, progress=True)
        net_list = list(resnext.children())
        self.layer0 = nn.Sequential(*net_list[:4])  # Initial convolution + BatchNorm + ReLU + MaxPool
        self.layer1 = net_list[4]  # First ResNeXt block
        self.layer2 = net_list[5]  # Second ResNeXt block
        self.layer3 = net_list[6]  # Third ResNeXt block
        self.layer4 = net_list[7]  # Fourth ResNeXt block

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #pdb.set_trace()
        return x0, x1, x2, x3, x4

class EDFModule(nn.Module):
    def __init__(self):
        super(EDFModule, self).__init__()
        self.low_level_edge_extractor = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64) # 修正: nn.Sequential 内に配置
        )

        self.edge_fusion = nn.Sequential(
            nn.Conv2d(64+512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, low_level_feat, low_level_feat2, high_level_feat, input_size):
        low_edge = self.low_level_edge_extractor(low_level_feat)
        high_edge = F.upsample(high_level_feat, size=low_edge.size()[2:], mode='bilinear', align_corners=True)
        #pdb.set_trace()

        fused_edge = self.edge_fusion(torch.cat((low_edge, high_edge), dim=1))
        return fused_edge

class EdgeDetectionNet(nn.Module):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()
        self.encoder = ResNeXtEncoder(pretrained=True)
        self.edf_module = EDFModule()

                # エンコーダ部分のパラメータを凍結
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x,y):
        input_size = x.shape[2:]  # Get the height and width of the input
        # Extract features from encoder
        _, low_level_feat, low_level_feat2, _, _= self.encoder(x)
        #pdb.set_trace()
        # Pass features through EDF module
        edge_map = self.edf_module(low_level_feat, low_level_feat2, y,input_size)
        edge_map = F.upsample(edge_map, size=input_size, mode='bilinear', align_corners=True)
        return edge_map


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = self.relu(x)
        return x


class RefinementNet(nn.Module):
    def __init__(self, in_c=10):
        super(RefinementNet, self).__init__()
        self.conv1 = BasicConv(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)

        self.final_conv = nn.Conv2d(64+10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #pdb.set_trace()

        fusion = self.conv1(x)
        fusion = self.conv2(fusion)
        fusion = self.res1(fusion)
        fusion = self.res2(fusion)
        fusion = self.res3(fusion)
        fusion = self.final_conv(torch.cat((fusion,x), dim=1))
        return fusion


'''
class DiscriminativeSubNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_channels=64):
        super(DiscriminativeSubNet, self).__init__()
        base_width = base_channels

        # Encoder 部分
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.enc_mp1 = nn.MaxPool2d(2)



        self.enc_block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.enc_mp2 = nn.MaxPool2d(2)

        self.enc_block3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.enc_mp3 = nn.MaxPool2d(2)

        self.enc_block4 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.enc_mp4 = nn.MaxPool2d(2)

        self.enc_block5 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.enc_mp5 = nn.MaxPool2d(2)

        self.enc_block6 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        # Decoder 部分
        self.dec_up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )
        self.dec_db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.dec_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )
        self.dec_db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.dec_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        self.dec_db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.dec_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.dec_db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.dec_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.dec_db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.dec_fin4_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )

        self.dec_fin3_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
        )
        self.dec_fin2_out = nn.Sequential(
            nn.Conv2d(base_width*2, out_channels, kernel_size=3, padding=1)
        )
        self.dec_fin1_out = nn.Sequential(
            nn.Conv2d(base_width* 4, out_channels, kernel_size=3, padding=1)
        )


    def forward(self, x):

        # U-Net部分の処理
        b1 = self.enc_block1(x)
        mp1 = self.enc_mp1(b1)
        b2 = self.enc_block2(mp1)
        mp2 = self.enc_mp2(b2)
        b3 = self.enc_block3(mp2)
        mp3 = self.enc_mp3(b3)
        b4 = self.enc_block4(mp3)
        mp4 = self.enc_mp4(b4)
        b5 = self.enc_block5(mp4)
        mp5 = self.enc_mp5(b5)
        b6 = self.enc_block6(mp5)

        # Decoder forward
        up_b = self.dec_up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.dec_db_b(cat_b)

        up1 = self.dec_up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.dec_db1(cat1)

        up2 = self.dec_up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.dec_db2(cat2)

        up3 = self.dec_up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.dec_db3(cat3)
        
        up4 = self.dec_up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.dec_db4(cat4)


        x_up1_1ch = self.dec_fin1_out(db1)
        x_up2_1ch = self.dec_fin2_out(db2)
        x_up3_1ch = self.dec_fin3_out(db3)
        x_up4_1ch = self.dec_fin4_out(db4)

        x_up1_1ch = F.interpolate(x_up1_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 
        x_up2_1ch = F.interpolate(x_up2_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 
        x_up3_1ch = F.interpolate(x_up3_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 
 

        #pdb.set_trace()
        return x_up1_1ch, x_up2_1ch, x_up3_1ch, x_up4_1ch,db1
'''
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.outconv=DoubleConv(in_channels + out_channels, out_channels, (in_channels + out_channels)// 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #pdb.set_trace()
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.outconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels/ 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        self.channelS = channelS
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        # Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        #pdb.set_trace()
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        V = self.value(S1)
        # Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z


class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x


class U_Transformer(nn.Module):
    def __init__(self, in_channels=5, classes = 1, base_width = 64,bilinear=True):
        super(U_Transformer, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_width)
        self.down1 = Down(base_width, 2 * base_width)
        self.down2 = Down(2 * base_width , 4 * base_width)
        self.down3 = Down(4 * base_width, 8 * base_width)
        self.MHSA = MultiHeadSelfAttention(8 * base_width)
        self.up1 = TransformerUp(8 * base_width, 4 * base_width)
        self.up2 = Up(4 * base_width, 2 * base_width)
        self.up3 = Up(2 * base_width, base_width)
        self.dec_fin4_out = nn.Sequential(
            nn.Conv2d(base_width, classes, kernel_size=3, padding=1)
        )

        self.dec_fin3_out = nn.Sequential(
            nn.Conv2d(base_width*2, classes, kernel_size=3, padding=1)
        )
        self.dec_fin2_out = nn.Sequential(
            nn.Conv2d(base_width*4, classes, kernel_size=3, padding=1)
        )
        self.dec_fin1_out = nn.Sequential(
            nn.Conv2d(base_width* 8, classes, kernel_size=3, padding=1)
        )
        self.outc = OutConv(base_width, classes)
        

    def forward(self, x):
        #pdb.set_trace()
        b1 = self.inc(x)
        b2 = self.down1(b1)
        b3 = self.down2(b2)
        b4 = self.down3(b3)
        m = self.MHSA(b4)
        #pdb.set_trace()
        u1 = self.up1(m, b3)
        u2 = self.up2(u1, b2)
        u3 = self.up3(u2, b1)
        x_up4_1ch = self.outc(u3)

        x_up1_1ch = self.dec_fin1_out(m)
        x_up2_1ch = self.dec_fin2_out(u1)
        x_up3_1ch = self.dec_fin3_out(u2)
        #x_up4_1ch = self.dec_fin4_out(u3)

        x_up1_1ch = F.interpolate(x_up1_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 
        x_up2_1ch = F.interpolate(x_up2_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 
        x_up3_1ch = F.interpolate(x_up3_1ch, size=x.size()[2:], mode='bilinear', align_corners=True) 

        return x_up1_1ch, x_up2_1ch, x_up3_1ch, x_up4_1ch,m

class MMM(nn.Module):

    def __init__(self):
        super(MMM, self).__init__()
        #self.DiscriminativeSubNet = DiscriminativeSubNet()
        self.DiscriminativeSubNet = U_Transformer()
        self.EdgeDetectionNet = EdgeDetectionNet()
        self.RefinementNet = nn.Conv2d(10, 1, 1, 1, 0)

    def forward(self, x, y, z, mode_flag):
        """
        mode_flag:
        1 -> Return features1, features2, features3, features4 (DiscriminativeSubNet)
        2 -> Return edge_out (EdgeDetectionNet)
        3 -> Return final_out (RefinementNet)
        """

        # Combine inputs
        input_cat = torch.cat([x,y, z], dim=1)

        # Discriminative SubNetwork
        features1, features2, features3, features4, db1 = self.DiscriminativeSubNet(input_cat)


        if mode_flag == 1:
            # Return features for mirror detection
            return features1, features2, features3, features4
            #return features4
        # Edge Detection
        edge_out = self.EdgeDetectionNet(x, db1)

        if mode_flag == 2:
            return edge_out

        # Refinement
        final_out = self.RefinementNet(
            torch.cat((features1, features2, features3, features4, edge_out, input_cat), dim=1)

        )
        if mode_flag == 3:
            return final_out
        else:
            return torch.sigmoid(final_out)