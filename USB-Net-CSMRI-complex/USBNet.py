import sys
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
import scipy.io as scio
from utils.utils import LayerNorm


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding=0),
        )
        self.se = SELayer(dim, reduction=16)

    def forward(self, x):
        short_cut = x
        x = self.block(x)
        x = self.se(x)
        x = x + short_cut
        return x


class IterationModule(nn.Module):
    def __init__(self, block_size, dim):
        super(IterationModule, self).__init__()
        self.block_size = block_size
        self.step = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.lamd = nn.Parameter(torch.tensor(1e-3), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(1e-0), requires_grad=True)
        self.real1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )
        self.imag1 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )

        self.unit1 = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, 2, kernel_size=1, padding=0),
        )
        self.unit2 = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            BasicBlock(dim),
            nn.Conv2d(dim, 2, kernel_size=1, padding=0),
        )

        self.real2 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )
        self.imag2 = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )

    def softshrink(self, x, mu):
        z = torch.zeros_like(x)
        x = torch.sign(x) * torch.maximum(torch.abs(x) - mu, z)
        return x

    def forward(self, mask, xk, dk, bk, x_init):
        ##########################STEP 1############################
        real, imag = torch.chunk(xk, 2, dim=1)
        xk_t1 = torch.complex(self.real1(real), self.imag1(imag))
        xk_t1 = torch.fft.fft2(xk_t1)

        xk_t1 = xk_t1 * mask.view(1, 1, *(mask.shape))

        xk_t1 = torch.fft.ifft2(xk_t1)
        xk_t1 = torch.cat([self.real2(xk_t1.real), self.imag2(xk_t1.imag)], dim=1)

        xk_t2 = self.unit2(self.unit1(xk) - (dk - bk))
        xk = xk - self.step * (xk_t1 - x_init) - self.step * self.mu * xk_t2

        ##########################STEP 2############################
        xk_c = self.unit1(xk)
        dk = self.softshrink(xk_c + bk, self.lamd / (self.mu + 1e-6))

        ##########################STEP 3############################
        bk = bk + xk_c - dk
        return xk, dk, bk


class UBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down1 = nn.Sequential(
            BasicBlock(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down2 = nn.Sequential(
            BasicBlock(2 * dim),
            nn.Conv2d(2 * dim, 4 * dim, kernel_size=2, stride=2),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.down3 = nn.Sequential(
            BasicBlock(4 * dim),
            nn.Conv2d(4 * dim, 8 * dim, kernel_size=2, stride=2),
            LayerNorm(8 * dim, eps=1e-6, data_format="channels_first"),
        )
        self.mid = nn.Sequential(
            BasicBlock(8 * dim),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8 * dim, out_channels=4 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(4 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(4 * dim),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4 * dim, out_channels=2 * dim, kernel_size=2, stride=2
            ),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2 * dim),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * dim, out_channels=dim, kernel_size=2, stride=2
            ),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )

    def forward(self, x):
        xk1 = self.down1(x)
        xk2 = self.down2(xk1)
        xk3 = self.down3(xk2)
        xk4 = self.mid(xk3)
        xk5 = self.up1(xk4) + xk2
        xk6 = self.up2(xk5) + xk1
        x = self.up3(xk6) + x
        return x


class USBNet(torch.nn.Module):
    def __init__(self, ratio, block_size=32, dim=32, depth=8):
        super(USBNet, self).__init__()
        self.ratio = ratio
        self.block_size = block_size
        self.depth = depth

        self.real = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )
        self.imag = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
        )

        self.pre = nn.Sequential(
            nn.Conv2d(2, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
        )
        self.iters = nn.ModuleList()
        self.blocks0 = nn.ModuleList()
        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        for i in range(self.depth):
            self.iters.append(IterationModule(self.block_size, dim=dim))
            self.blocks0.append(
                nn.Sequential(
                    nn.Conv2d(2, dim, kernel_size=3, padding=1),
                    BasicBlock(dim),
                )
            )
            self.blocks1.append(
                nn.Sequential(
                    UBlock(dim),
                )
            )
            self.blocks2.append(
                nn.Sequential(
                    LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                    BasicBlock(dim),
                    nn.Conv2d(dim, 2, kernel_size=3, padding=1),
                )
            )

        self.mid = nn.Sequential(
            nn.Conv2d(2 * self.depth, dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(dim),
        )
        self.post = nn.Sequential(
            UBlock(2 * dim),
            LayerNorm(2 * dim, eps=1e-6, data_format="channels_first"),
            BasicBlock(2 * dim),
            nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1),
            BasicBlock(dim),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )
        self.apply(self._init_weights)

    def forward(self, x, mask):
        # Sampling
        x_k_sapce = torch.fft.fft2(x)
        y = x_k_sapce * mask.view(1, 1, *(mask.shape))

        # init
        x_init = torch.fft.ifft2(y)
        x_init = torch.cat([self.real(x_init.real), self.imag(x_init.imag)], dim=1)

        # Recon
        dk, bk = torch.zeros_like(x_init), torch.zeros_like(x_init)
        xk_i = x_init
        inter_xk = []
        xk = self.pre(x_init)
        for i in range(self.depth):
            xk_i, dk, bk = self.iters[i](mask, xk_i, dk, bk, x_init)
            xk = xk + self.blocks0[i](xk_i)
            xk = self.blocks1[i](xk)
            xk_i = self.blocks2[i](xk)
            inter_xk.append(xk_i)

        inter_xk = torch.cat(inter_xk, dim=1)
        inter_xk = self.mid(inter_xk)
        xk = torch.cat([xk, inter_xk], dim=1)
        xk = self.post(xk)

        return xk

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model(ratio, model_name, mask_type):
    path = f"./model-{mask_type}/checkpoint-{model_name}-{ratio}-best.pth"
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def load_sampling_matrix(ratio, image_size, mask_type):
    dir = f"../data/mask/{image_size}/{mask_type}_{ratio}.mat"
    if mask_type == "Cartesian":
        mask = scio.loadmat(dir)["mask"]
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
    elif mask_type == "Radial":
        mask = scio.loadmat(dir)["mask_matrix"]
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return mask


@register_model
def usbnet(ratio, pretrained=False, mask_type="Cartesian", **kwargs):
    model = USBNet(ratio, block_size=32, dim=32, depth=8)
    if pretrained:
        checkpoint = load_pretrained_model(
            ratio, sys._getframe().f_code.co_name, mask_type
        )
        model.load_state_dict(checkpoint["model"])
    return model
