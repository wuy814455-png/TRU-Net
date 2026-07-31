import torch
import torch.nn as nn

def apply_2d_module_over_time(module: nn.Module, x: torch.Tensor):
    if x.dim() == 4:
        return module(x)
    if x.dim() != 5:
        raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')
    b, t, c, h, w = x.shape
    y = module(x.reshape(b * t, c, h, w))
    if y.dim() != 4:
        raise RuntimeError('The wrapped module must return a 4D tensor.')
    y = y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])
    return y

def center_crop_like(src, target):
    _, _, H, W = src.shape
    _, _, h, w = target.shape
    top = (H - h) // 2
    left = (W - w) // 2
    return src[:, :, top:top + h, left:left + w]

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class ModifiedUNetTD(nn.Module):

    def __init__(self, inputchannel=6, outputchannel=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(inputchannel, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.final_upconv = nn.Sequential(nn.ConvTranspose2d(32, 32, kernel_size=5, stride=5, output_padding=1), nn.Conv2d(32, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.Conv2d(32, outputchannel, kernel_size=3, padding=1))

    def _forward_4d(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.upconv3(e3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.upconv2(d3)
        e1_crop = center_crop_like(e1, d2)
        d2 = self.dec2(torch.cat([d2, e1_crop], dim=1))
        out = self.final_upconv(d2)
        out = out[:, :, :161, :161]
        return out

    def _forward_5d(self, x):
        e1 = apply_2d_module_over_time(self.enc1, x)
        e2 = apply_2d_module_over_time(self.enc2, apply_2d_module_over_time(self.pool, e1))
        e3 = apply_2d_module_over_time(self.enc3, apply_2d_module_over_time(self.pool, e2))
        d3 = apply_2d_module_over_time(self.upconv3, e3)
        d3 = apply_2d_module_over_time(self.dec3, torch.cat([d3, e2], dim=2))
        d2 = apply_2d_module_over_time(self.upconv2, d3)
        b, t, c, h, w = d2.shape
        e1_reshape = e1.reshape(b * t, e1.shape[2], e1.shape[3], e1.shape[4])
        d2_reshape = d2.reshape(b * t, d2.shape[2], d2.shape[3], d2.shape[4])
        e1_crop = center_crop_like(e1_reshape, d2_reshape)
        e1_crop = e1_crop.reshape(b, t, e1_crop.shape[1], e1_crop.shape[2], e1_crop.shape[3])
        d2 = apply_2d_module_over_time(self.dec2, torch.cat([d2, e1_crop], dim=2))
        out = apply_2d_module_over_time(self.final_upconv, d2)
        out = out[..., :161, :161]
        return out

    def forward(self, x):
        if x.dim() == 4:
            return self._forward_4d(x)
        if x.dim() == 5:
            return self._forward_5d(x)
        raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')
