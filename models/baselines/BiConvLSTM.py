import torch
import torch.nn as nn

def ensure_5d(x: torch.Tensor):
    if x.dim() == 4:
        return (x.unsqueeze(1), True)
    if x.dim() == 5:
        return (x, False)
    raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')

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

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding, bias=bias)

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
        return (h_next, c_next)

    def init_state(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return (h, c)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class BiConvLSTMLayer(nn.Module):

    def __init__(self, in_channels=8, hidden_dim=8, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.convin = ConvBlock(in_channels, hidden_dim)
        self.forward_cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True)
        self.backward_cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x5, squeezed = ensure_5d(x)
        x5 = apply_2d_module_over_time(self.convin, x5)
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
        out = torch.stack(fused_seq, dim=1)
        if squeezed:
            return out.squeeze(1)
        return out

class BiConvLSTM(nn.Module):

    def __init__(self, inputchannel=6, outputchannel=1, hidden_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inputchannel = inputchannel
        self.met_channels = inputchannel - 1
        self.layer1 = BiConvLSTMLayer(in_channels=self.met_channels, hidden_dim=hidden_dim, kernel_size=3)
        self.layer2 = BiConvLSTMLayer(in_channels=hidden_dim, hidden_dim=hidden_dim, kernel_size=3)
        self.terrain_branch = nn.Sequential(nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(hidden_dim * 2), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.upsample_head = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim * 25, kernel_size=3, padding=1), nn.PixelShuffle(5), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_dim // 2, outputchannel, kernel_size=3, padding=1))

    def forward(self, x, terrain=None):
        if x.dim() == 4:
            b, c, h, w = x.shape
        elif x.dim() == 5:
            b, t, c, h, w = x.shape
        else:
            raise ValueError(f'Unsupported input shape: {tuple(x.shape)}')
        if x.dim() == 4:
            met = x[:, :self.met_channels, :, :]
            terrain_33 = x[:, self.met_channels:, :, :]
            met_seq = met.unsqueeze(1)
        else:
            met_seq = x[:, :, :self.met_channels, :, :]
            terrain_33 = x[:, 0, self.met_channels:, :, :]
        feat = self.layer1(met_seq)
        feat = self.layer2(feat)
        terrain_feat = self.terrain_branch(terrain_33)
        feat = feat[:, -1]
        feat = torch.cat([feat, terrain_feat], dim=1)
        feat = self.conv_out(feat)
        out = self.upsample_head(feat)
        out = out[:, :, :161, :161]
        return out
