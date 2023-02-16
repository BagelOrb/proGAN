import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):
    """Weight scaled convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super(ConvBlock, self).__init__()
        self.use_pixel_norm = use_pixel_norm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        if self.use_pixel_norm:
            x = self.pn(x)
        x = self.leaky(self.conv2(x))
        if self.use_pixel_norm:
            x = self.pn(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 1x1 -> 4x4 (kernel_size, stride, padding)
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )  # TODO: implement weight scaling for this first block

        self.initial_to_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, padding=0)

        self.prog_blocks, self.to_rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_to_rgb]),
        )
        # TODO: split rgb_layers into two: one for upscaled and one for upscaled+convoluted

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.to_rgb_layers.append(WSConv2d(conv_out_channels, img_channels, kernel_size=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return alpha * generated + (1 - alpha) * upscaled

    def forward(self, x, alpha, n_blocks):
        out = self.initial(x)

        if n_blocks == 0:  # TODO: move to below the following loop and dissipate
            return self.initial_to_rgb(out)  # TODO: replace by rgb_layers[0]

        for i_block in range(n_blocks):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[i_block](upscaled)

        final_upscaled = self.to_rgb_layers[n_blocks - 1](upscaled)
        # reuses toRGB from the previous layer because it's the same mapping from channels to rgb
        # The dimensionality change doesn't matter because it's a 1x1 conv kernel
        final_out = self.to_rgb_layers[n_blocks](out)
        return torch.tanh(self.fade_in(alpha, final_upscaled, final_out))


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.from_rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):  # (excluding zero)
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.from_rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, padding=0))

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # for img of 4x4
        self.initial_from_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, padding=0)
        self.from_rgb_layers.append(self.initial_from_rgb)
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0),
            # 0 padding so that output size is 1x1 img
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0)
            # TODO: officially this layer shouldn't have had the ReLU act function
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        # TODO: is pytorch backpropagating through this formula?!
        return torch.cat([x, batch_stats], dim=1)  # 512 -> 513

    def forward(self, x, alpha, n_blocks):

        starting_block_i = len(self.prog_blocks) - n_blocks
        out = self.leaky(self.from_rgb_layers[starting_block_i](x))

        if n_blocks == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.from_rgb_layers[starting_block_i + 1](self.avg_pool(x)))

        out = self.avg_pool(self.prog_blocks[starting_block_i](out))
        out = self.fade_in(alpha, downscaled, out)

        for i_block in range(starting_block_i + 1, len(self.prog_blocks)):
            out = self.prog_blocks[i_block](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for num_blocks in range(9):
        img_size = 2 ** (num_blocks + 2)
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, .5, n_blocks=num_blocks)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=.5, n_blocks=num_blocks)
        assert out.shape == (1, 1)
        print(f"succes at img size {img_size}")