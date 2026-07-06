import torch
import torch.nn as nn

class DoubleConv(nn.Module):
  
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, inputchannel=5, outputchannel=1, BN_enable=True):
        super(UNet, self).__init__()
        self.BN_enable = BN_enable
      
        self.enc1 = DoubleConv(inputchannel, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
      
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1) 
        self.dec1 = DoubleConv(128, 64)
       
        self.final_upconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=5, output_padding=0), 
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  
            nn.Conv2d(64, outputchannel, kernel_size=5, padding=2), 
        )

    def forward(self, x, terrain=None):
    
        e1 = self.enc1(x)  # 64, 33x33
        e2 = self.enc2(self.pool(e1))  
        e3 = self.enc3(self.pool(e2)) 
        e4 = self.enc4(self.pool(e3))  
       
        b = self.bottleneck(self.pool(e4))  
       
        d4 = self.upconv4(b)  # 512, 4x4
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  
        d3 = self.upconv3(d4)  # 256, 8x8
        d3 = self.dec3(torch.cat([d3, e3], dim=1)) 
        d2 = self.upconv2(d3)  # 128, 16x16
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  
        d1 = self.upconv1(d2)  # 64, 33x33
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  
      
        out = self.final_upconv(d1)[:, :, :161, :161] 
        return out

