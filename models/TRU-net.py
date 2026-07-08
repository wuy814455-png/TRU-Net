#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torchvision.models import resnet34  # Modified: Changed from res2net50 to resnet34
import os


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='transpose', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(
                in_channels=mid_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False
            )
        else:
            self.upsample = nn.PixelShuffle(upscale_factor=2)
            if mid_channels != out_channels * 4:
                self.conv_adjust = nn.Conv2d(mid_channels, out_channels * 4, kernel_size=1, bias=False)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv1(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = tnf.relu(x)
        if self.upsample_mode == 'transpose':
            x = self.upsample(x)
        else:
            if hasattr(self, 'conv_adjust'):
                x = self.conv_adjust(x)
            x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = tnf.relu(x)
        attention = self.spatial_attention(x)
        x = x * attention
        return x

class TerrainFeatureExtractor(nn.Module):
   
    def __init__(self, BN_enable=True):
        super().__init__()
        self.BN_enable = BN_enable
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if BN_enable else nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool2d((33, 33)) 
       
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128) if BN_enable else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d((17, 17)) 
       
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256) if BN_enable else nn.Identity()
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((9, 9))  
    
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512) if BN_enable else nn.Identity()
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool2d((5, 5))  
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024) if BN_enable else nn.Identity()
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AdaptiveAvgPool2d((5, 5)) 
   
        self.adjust1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)   # 匹配 encoder1 (64)
        self.adjust2 = nn.Conv2d(128, 128, kernel_size=1, bias=False) # 匹配 encoder2 (128)
        self.adjust3 = nn.Conv2d(256, 256, kernel_size=1, bias=False) # 匹配 encoder3 (256)
        self.adjust4 = nn.Conv2d(512, 512, kernel_size=1, bias=False) # 匹配 encoder4 (512)
        self.adjust5 = nn.Conv2d(1024, 512, kernel_size=1, bias=False) # 匹配 center (512)

    def forward(self, x):
        # 960x960 → 480x480 → 33x33
        t1 = self.conv1(x)
        t1 = self.bn1(t1)
        t1 = self.relu1(t1)
        t1_out = self.pool1(t1)  
        t1_out = self.adjust1(t1_out) 
        
        t2 = self.conv2(t1)
        t2 = self.bn2(t2)
        t2 = self.relu2(t2)
        t2_out = self.pool2(t2) 
        t2_out = self.adjust2(t2_out)  
     
        t3 = self.conv3(t2)
        t3 = self.bn3(t3)
        t3 = self.relu3(t3)
        t3_out = self.pool3(t3) 
        t3_out = self.adjust3(t3_out)  
        
        t4 = self.conv4(t3)
        t4 = self.bn4(t4)
        t4 = self.relu4(t4)
        t4_out = self.pool4(t4)  
        t4_out = self.adjust4(t4_out)  
  
        t5 = self.conv5(t4)
        t5 = self.bn5(t5)
        t5 = self.relu5(t5)
        t5_out = self.pool5(t5)  # [batch_size, 1024, 5, 5]
        t5_out = self.adjust5(t5_out)  # [batch_size, 512, 5, 5]
        return [t1_out, t2_out, t3_out, t4_out, t5_out]

class Res34_Unet(nn.Module):  # Modified: Renamed from Res2Net50_Unet to Res34_Unet
    def __init__(self, inputchannel, outputchannel, BN_enable=True, resnet_pretrain=True, output_size=(161, 161)):
        super().__init__()
        self.BN_enable = BN_enable
        self.resnet = resnet34(pretrained=resnet_pretrain)  # Modified: Changed to resnet34
        filters = [64, 64, 128, 256, 512]  # Modified: Adjusted filters to match ResNet34
        self.output_size = output_size
    
        self.firstconv = nn.Conv2d(in_channels=inputchannel, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.firstbn = self.resnet.bn1
        self.firstrelu = tnf.relu
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4
       
        self.terrain_extractor = TerrainFeatureExtractor(BN_enable=BN_enable)
      (Modified: Adjusted in_channels and mid_channels to match ResNet34 filters)
        self.center = DecoderBlock(in_channels=filters[4], mid_channels=filters[4] * 4, out_channels=filters[4],
                                   upsample_mode='transpose', BN_enable=self.BN_enable)
        self.decoder0 = DecoderBlock(in_channels=filters[4] + filters[3], mid_channels=filters[3] * 4,
                                     out_channels=filters[3], upsample_mode='transpose', BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], upsample_mode='transpose', BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], upsample_mode='transpose', BN_enable=self.BN_enable)
        self.dropout = nn.Dropout(p=0.3)
        
        self.final_upsample = nn.Sequential(
            nn.Conv2d(in_channels=filters[1], out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128) if BN_enable else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64) if BN_enable else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32) if BN_enable else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32) if BN_enable else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=outputchannel, kernel_size=3, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(output_size)
        )
        if resnet_pretrain:
           
            state_dict = self.resnet.state_dict()
            if inputchannel != 3:
                pretrained_conv1_weight = state_dict['conv1.weight']
                new_conv1_weight = torch.zeros(64, inputchannel, 7, 7, device=pretrained_conv1_weight.device)
                for i in range(inputchannel):
                    if i < 3:
                        new_conv1_weight[:, i, :, :] = pretrained_conv1_weight[:, i, :, :]
                    else:
                        new_conv1_weight[:, i, :, :] = pretrained_conv1_weight[:, :3, :, :].mean(dim=1)
                state_dict['conv1.weight'] = new_conv1_weight
            self.firstconv.weight.data = state_dict['conv1.weight']
            print("ResNet34 pretrained weights loaded, adapted for input channels:", inputchannel)

    def forward(self, x, terrain):
      
        terrain_features = self.terrain_extractor(terrain)  # [t1, t2, t3, t4, t5]

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        e1 = self.encoder1(x_) + terrain_features[0]  
        e2 = self.encoder2(e1) + terrain_features[1]
        e3 = self.encoder3(e2) + terrain_features[2]
        e4 = self.encoder4(e3) + terrain_features[3]
        center = self.center(e4 + terrain_features[4])
        center = tnf.interpolate(center, size=(e3.size(2), e3.size(3)), mode='bilinear', align_corners=False)
        d1 = self.decoder0(torch.cat([center, e3], dim=1))
        d1 = tnf.interpolate(d1, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=False)
        d2 = self.decoder1(torch.cat([d1, e2], dim=1))
        d2 = tnf.interpolate(d2, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=False)
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d3 = self.dropout(d3)
        output = self.final_upsample(d3)
        return output

class ModifiedRes34_Unet(Res34_Unet):
    def __init__(self, inputchannel, outputchannel, BN_enable=True, resnet_pretrain=True, output_size=(161, 161)):
        super().__init__(inputchannel, outputchannel, BN_enable=BN_enable, resnet_pretrain=resnet_pretrain, output_size=output_size)

if __name__ == "__main__":

    Model = ModifiedRes34_Unet(inputchannel=6, outputchannel=1, BN_enable=True, resnet_pretrain=True, output_size=(161, 161))
    a = torch.rand((5, 6, 33, 33))
    terrain = torch.rand((5, 1, 960, 960))
    c = Model(a, terrain)
    print(f"Output shape: {c.shape}")  
