#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def ensure_5d(x: torch.Tensor):
   
    if x.dim() == 4:
        return x.unsqueeze(1), True
    if x.dim() == 5:
        return x, False
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def apply_2d_module_over_time(module: nn.Module, x: torch.Tensor):
  
    if x.dim() == 4:
        return module(x)

    if x.dim() != 5:
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    b, t, c, h, w = x.shape
    y = module(x.reshape(b * t, c, h, w))
    if y.dim() != 4:
        raise RuntimeError("The wrapped module must return a 4D tensor.")
    y = y.reshape(b, t, y.shape[1], y.shape[2], y.shape[3])
    return y

def center_crop_like(src, target):
  
    _, _, H, W = src.shape
    _, _, h, w = target.shape

    top = (H - h) // 2
    left = (W - w) // 2

    return src[
        :,
        :,
        top:top + h,
        left:left + w
    ]


class DoubleConv(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ModifiedUNetTD(nn.Module):
      def __init__(
            self,
            inputchannel=6,
            outputchannel=1):

        super().__init__()

       
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        
        self.enc1 = DoubleConv(
            inputchannel,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

       
        self.upconv3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            128,
            64
        )

        self.upconv2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            64,
            32
        )

      
        self.final_upconv = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                32,
                kernel_size=5,
                stride=5,
                output_padding=1
            ),

            nn.Conv2d(
                32,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                outputchannel,
                kernel_size=3,
                padding=1
            )
        )

  
    def _forward_4d(self, x):

       
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )                                

        e3 = self.enc3(
            self.pool(e2)
        )                                 

      
        d3 = self.upconv3(e3)               

        d3 = self.dec3(
            torch.cat(
                [d3, e2],
                dim=1
            )
        )

        d2 = self.upconv2(d3)                

        e1_crop = center_crop_like(
            e1,
            d2
        )

        d2 = self.dec2(
            torch.cat(
                [d2, e1_crop],
                dim=1
            )
        )

        out = self.final_upconv(d2)

        out = out[:, :, :161, :161]

        return out

  
    def _forward_5d(self, x):

        e1 = apply_2d_module_over_time(
            self.enc1,
            x
        )

        e2 = apply_2d_module_over_time(
            self.enc2,
            apply_2d_module_over_time(
                self.pool,
                e1
            )
        )

        e3 = apply_2d_module_over_time(
            self.enc3,
            apply_2d_module_over_time(
                self.pool,
                e2
            )
        )

        d3 = apply_2d_module_over_time(
            self.upconv3,
            e3
        )

        d3 = apply_2d_module_over_time(
            self.dec3,
            torch.cat(
                [d3, e2],
                dim=2
            )
        )

        d2 = apply_2d_module_over_time(
            self.upconv2,
            d3
        )

        
        b, t, c, h, w = d2.shape

        e1_reshape = e1.reshape(
            b * t,
            e1.shape[2],
            e1.shape[3],
            e1.shape[4]
        )

        d2_reshape = d2.reshape(
            b * t,
            d2.shape[2],
            d2.shape[3],
            d2.shape[4]
        )

        e1_crop = center_crop_like(
            e1_reshape,
            d2_reshape
        )

        e1_crop = e1_crop.reshape(
            b,
            t,
            e1_crop.shape[1],
            e1_crop.shape[2],
            e1_crop.shape[3]
        )

        d2 = apply_2d_module_over_time(
            self.dec2,
            torch.cat(
                [d2, e1_crop],
                dim=2
            )
        )

        out = apply_2d_module_over_time(
            self.final_upconv,
            d2
        )

        out = out[..., :161, :161]

        return out

 
    def forward(self, x):

        if x.dim() == 4:
            return self._forward_4d(x)

        if x.dim() == 5:
            return self._forward_5d(x)

        raise ValueError(
            f"Unsupported input shape: {tuple(x.shape)}"
        )



class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = kernel_size // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvBlock(nn.Module):
   
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BiConvLSTMLayer(nn.Module):
   
    def __init__(self, in_channels=8, hidden_dim=8, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 论文中的 Convin：两层 3×3 Conv + ReLU
        self.convin = ConvBlock(in_channels, hidden_dim)

        self.forward_cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True)
        self.backward_cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True)

    def forward(self, x):
       
        x5, squeezed = ensure_5d(x)   # (B, T, C, H, W)
        x5 = apply_2d_module_over_time(self.convin, x5)  # (B, T, hidden_dim, H, W)

        b, t, c, h, w = x5.shape

        
        h_f, c_f = self.forward_cell.init_state(b, h, w, x5.device)
        forward_states = []
        for ti in range(t):
            h_f, c_f = self.forward_cell(x5[:, ti], h_f, c_f)
            forward_states.append(h_f)

       
        h_b, c_b = self.backward_cell.init_state(b, h, w, x5.device)
        backward_states = []
        for ti in reversed(range(t)):
            h_b, c_b = self.backward_cell(x5[:, ti], h_b, c_b)
            backward_states.append(h_b)
        backward_states = backward_states[::-1]

        
        fused_seq = []
        for ti in range(t):
            fused_seq.append(x5[:, ti] + forward_states[ti] + backward_states[ti])

        out = torch.stack(fused_seq, dim=1)  # (B, T, hidden_dim, H, W)

        if squeezed:
            return out.squeeze(1)  # (B, hidden_dim, H, W)
        return out


class BiConvLSTM(nn.Module):
    
    def __init__(self, inputchannel=6, outputchannel=1, hidden_dim=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.inputchannel = inputchannel

      
        self.met_channels = inputchannel - 1   

       
        self.layer1 = BiConvLSTMLayer(
            in_channels=self.met_channels,
            hidden_dim=hidden_dim,
            kernel_size=3
        )
        self.layer2 = BiConvLSTMLayer(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=3
        )

        
        self.terrain_branch = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        
        self.upsample_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 25, kernel_size=3, padding=1),
            nn.PixelShuffle(5),                                    
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, outputchannel, kernel_size=3, padding=1),
        )

    def forward(self, x, terrain=None):
       
        if x.dim() == 4:
            b, c, h, w = x.shape
        elif x.dim() == 5:
            b, t, c, h, w = x.shape
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

       
        if x.dim() == 4:
            met = x[:, :self.met_channels, :, :]        
            terrain_33 = x[:, self.met_channels:, :, :] 
            met_seq = met.unsqueeze(1)                    
        else:
            met_seq = x[:, :, :self.met_channels, :, :]  
            terrain_33 = x[:, 0, self.met_channels:, :, :]  

      
        feat = self.layer1(met_seq)   # (B, T, hidden_dim, H, W)
        feat = self.layer2(feat)      # (B, T, hidden_dim, H, W)

       
        terrain_feat = self.terrain_branch(terrain_33)  
        feat = feat + terrain_feat.unsqueeze(1)         

       
        feat = feat[:, -1]   # (B, hidden_dim, H, W)

       
        feat = self.conv_out(feat)   # (B, hidden_dim, H, W)

      
        out = self.upsample_head(feat)   # (B, 1, 165, 165)
        out = out[:, :, :161, :161]      # (B, 1, 161, 161)

        return out
