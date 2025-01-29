import torch
import torch.nn as nn
import torch.nn.functional as F


import pdb

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
        '''
        self.low_level_edge_extractor2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64) # 修正: nn.Sequential 内に配置
        )
        '''
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(64+256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, low_level_feat, low_level_feat2, high_level_feat, input_size):
        low_edge = self.low_level_edge_extractor(low_level_feat)
        #low_edge2 = self.low_level_edge_extractor2(low_level_feat2)
        #low_edge2 = F.upsample(low_edge2, size=low_edge.size()[2:], mode='bilinear', align_corners=True)
        high_edge = F.upsample(high_level_feat, size=low_edge.size()[2:], mode='bilinear', align_corners=True)
        #pdb.set_trace()
        fused_edge = self.edge_fusion(torch.cat((low_edge, high_edge), dim=1))
        return fused_edge

class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
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
    def __init__(self, in_c=5):
        super(RefinementNet, self).__init__()
        self.conv1 = BasicConv(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)

        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)

        self.final_conv = nn.Conv2d(64+5, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, image, d_image, d_image_2,saliency_map1,saliency_map2,saliency_map3,saliency_map4, edge):
        #pdb.set_trace()
        fusion_0 = torch.cat((saliency_map1,saliency_map2,saliency_map3,saliency_map4, edge), dim=1)
        fusion = self.conv1(fusion_0)
        fusion = self.conv2(fusion)
        fusion = self.res1(fusion)
        fusion = self.res2(fusion)
        fusion = self.res3(fusion)
        fusion = self.final_conv(torch.cat((fusion,fusion_0), dim=1))
        return fusion



class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base_channels=64):
        super(DiscriminativeSubNetwork, self).__init__()
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



class MMM(nn.Module):

    def __init__(self):
        super(MMM, self).__init__()
        self.DiscriminativeSubNetwork = DiscriminativeSubNetwork()
        self.EdgeDetectionModel = EdgeDetectionModel()
        self.refinement_net = nn.Conv2d(10, 1, 1, 1, 0)

    def forward(self, x, y, z, mode_flag):
        """
        mode_flag:
        1 -> Return features1, features2, features3, features4 (mirror detection)
        2 -> Return edge_out (Edge Detection)
        3 -> Return final_out (Refinement)
        """

        # Combine inputs
        input_cat = torch.cat([x, y, z], dim=1)

        # Discriminative SubNetwork
        features1, features2, features3, features4, db1 = self.DiscriminativeSubNetwork(input_cat)


        if mode_flag == 1:
            # Return features for mirror detection
            return features1, features2, features3, features4
        # Edge Detection
        edge_out = self.EdgeDetectionModel(x, db1)

        if mode_flag == 2:
            return edge_out

        # Refinement
        final_out = self.refinement_net(
            torch.cat((features1, features2, features3, features4, edge_out, input_cat), dim=1)

        )
        if mode_flag == 3:
            return final_out
        else:
            return torch.sigmoid(final_out)