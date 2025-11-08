import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def _center_crop_to(x, H, W):
    h, w = x.shape[-2], x.shape[-1]
    if h > H:
        top = (h - H) // 2
        x = x[..., top:top + H, :]
    if w > W:
        left = (w - W) // 2
        x = x[..., :, left:left + W]
    return x

def _maybe_depthwise_conv(in_ch, out_ch, k=3, padding=1, use_dw=True):
    if not use_dw:
        return nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=False)
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, padding=padding, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
    )

def _conv_bn_relu(in_ch, out_ch, k=3, padding=1, use_dw=True):
    return nn.Sequential(
        _maybe_depthwise_conv(in_ch, out_ch, k=k, padding=padding, use_dw=use_dw),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, padding=1):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=padding)

    def forward(self, x, state):
        h, c = state
        z = torch.cat([x, h], dim=1)
        gates = self.conv(z)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def init_state(self, b, h, w, device):
        ch = self.hid_ch
        return (torch.zeros(b, ch, h, w, device=device),
                torch.zeros(b, ch, h, w, device=device))

class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid_ch=(16, 16)):
        super().__init__()
        self.cells = nn.ModuleList()
        ch_in = in_ch
        for ch in hid_ch:
            self.cells.append(ConvLSTMCell(ch_in, ch))
            ch_in = ch

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        states = [cell.init_state(B, H, W, x_seq.device) for cell in self.cells]
        out = None
        for t in range(T):
            x = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                h, states[i] = cell(x, states[i])
                x = h
            out = x
        return out  # [B, hidden, H, W]

class UNetHead(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=16, use_depthwise=None):
        super().__init__()
        if use_depthwise is None:
            use_depthwise = (os.environ.get("WAVE_USE_DEPTHWISE", "1") != "0")
        self.use_dw = use_depthwise
        b = base

        # Encoder
        self.enc1 = nn.Sequential(
            _conv_bn_relu(in_ch, b, use_dw=self.use_dw),
            _conv_bn_relu(b, b, use_dw=self.use_dw),
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            _conv_bn_relu(b, 2*b, use_dw=self.use_dw),
            _conv_bn_relu(2*b, 2*b, use_dw=self.use_dw),
        )
        self.down2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bott = nn.Sequential(
            _conv_bn_relu(2*b, 2*b, use_dw=self.use_dw),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(2*b, b, 2, stride=2)
        self.dec2 = nn.Sequential(
            _conv_bn_relu(3*b, b, use_dw=self.use_dw),
            _conv_bn_relu(b, b, use_dw=self.use_dw),
        )

        self.up1 = nn.ConvTranspose2d(b, b, 2, stride=2)
        self.dec1 = nn.Sequential(
            _conv_bn_relu(2*b, b, use_dw=self.use_dw),
            _conv_bn_relu(b, b, use_dw=self.use_dw),
        )

        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        H0, W0 = x.shape[-2], x.shape[-1]

        e1 = self.enc1(x)
        x  = self.down1(e1)
        e2 = self.enc2(x)
        x  = self.down2(e2)

        x  = self.bott(x)

        x  = self.up2(x)
        e2c = _center_crop_to(e2, x.shape[-2], x.shape[-1])
        x  = torch.cat([x, e2c], dim=1)
        x  = self.dec2(x)

        x  = self.up1(x)
        e1c = _center_crop_to(e1, x.shape[-2], x.shape[-1])
        x  = torch.cat([x, e1c], dim=1)
        x  = self.dec1(x)

        y = self.out(x)
        if y.shape[-2:] != (H0, W0):
            y = F.interpolate(y, size=(H0, W0), mode="bilinear", align_corners=False)
        return y

class WavePredictor(nn.Module):
    def __init__(self, T_in=6, K=1):
        super().__init__()
        self.encoder = ConvLSTM(in_ch=1, hid_ch=(16, 16))
        self.head = UNetHead(in_ch=16, out_ch=K, base=16)

    def forward(self, x_seq):
        h = self.encoder(x_seq)
        y = self.head(h)
        return y.unsqueeze(2)
