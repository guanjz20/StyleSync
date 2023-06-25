import math
import itertools
import cv2
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from op.fused_act import FusedLeakyReLU
from op.fused_act import fused_leaky_relu
from op.upfirdn2d import upfirdn2d

isconcat = True


def make_kernel(k):
    k = paddle.to_tensor(k, dtype='float32')
    if k.ndim == 1:
        k = k.unsqueeze(0) * k.unsqueeze(1)
    k /= k.sum()
    return k


class PixelNorm(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input):

        return input * paddle.rsqrt(paddle.mean(input * input, axis=1, keepdim=True) + 1e-8)


class Upfirdn2dUpsample(nn.Layer):

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor * factor)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Upfirdn2dDownsample(nn.Layer):

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Upfirdn2dBlur(nn.Layer):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor * upsample_factor)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2D(nn.Layer):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = self.create_parameter((out_channel, in_channel, kernel_size, kernel_size),
                                            default_initializer=nn.initializer.Normal())
        self.scale = 1 / math.sqrt(in_channel * (kernel_size * kernel_size))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = self.create_parameter((out_channel,), nn.initializer.Constant(0.0))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return ("{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
                " {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})")


class EqualLinear(nn.Layer):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = self.create_parameter((in_dim, out_dim), default_initializer=nn.initializer.Normal())
        self.weight.set_value((self.weight / lr_mul))
        if bias:
            self.bias = self.create_parameter((out_dim,), nn.initializer.Constant(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return ("{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]})")


class ScaledLeakyReLU(nn.Layer):

    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2D(nn.Layer):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * (kernel_size * kernel_size)
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = self.create_parameter((1, out_channel, in_channel, kernel_size, kernel_size),
                                            default_initializer=nn.initializer.Normal())

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return ("{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
                "upsample={self.upsample}, downsample={self.downsample})")

    def forward(self, in_im, style):
        batch, in_channel, height, width = in_im.shape
        style = self.modulation(style)
        style = style.reshape((batch, 1, in_channel, 1, 1))
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = paddle.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape((batch, self.out_channel, 1, 1, 1))

        weight = weight.reshape((batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size))

        if self.upsample:
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            weight = weight.reshape((batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size))
            weight = weight.transpose((0, 2, 1, 3, 4))
            weight = weight.reshape((batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size))
            out = F.conv2d_transpose(in_im, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))
            out = self.blur(out)

        elif self.downsample:
            in_im = self.blur(in_im)
            _, _, height, width = in_im.shape
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(in_im, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        else:
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(in_im, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        return out


class NoiseInjection(nn.Layer):

    def __init__(self):
        super().__init__()

        self.weight = self.create_parameter((1,), default_initializer=nn.initializer.Constant(0.0))

    def forward(self, image, noise=None):
        if noise is not None:
            if isconcat:
                return paddle.concat([image, self.weight * noise], 1)
            return image + self.weight * noise

        if noise is None:
            batch, _, height, width = image.shape
            noise = paddle.randn((batch, 1, height, width))

        if isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image * (1 - self.weight) + self.weight * noise


class ConstantInput(nn.Layer):

    def __init__(self, channel, size=4):
        super().__init__()

        self.input = self.create_parameter((1, channel, size, size), default_initializer=nn.initializer.Normal())

    def forward(self, input):

        batch = input.shape[0]
        out = self.input.tile((batch, 1, 1, 1))

        return out


class StyledConv(nn.Layer):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2D(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel * feat_multiplier)

    def forward(self, input, style, noise=None):

        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Layer):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upfirdn2dUpsample(blur_kernel)

        self.conv = ModulatedConv2D(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = self.create_parameter((1, 3, 1, 1), nn.initializer.Constant(0.0))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvLayer(nn.Sequential):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2D(
            in_channel,
            out_channel,
            kernel_size,
            padding=self.padding,
            stride=stride,
            bias=bias and not activate,
        ))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class audioConv2d(nn.Layer):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cout >= 64:
            goalSize = 16
        else:
            goalSize = 8
        self.conv_block = nn.Sequential(nn.Conv2D(cin, cout, kernel_size, stride, padding),
                                        nn.GroupNorm(num_groups=goalSize, num_channels=cout))
        self.act = nn.LeakyReLU(0.01)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Layer):

    def __init__(self, lr_mlp=0.01):
        super().__init__()
        self.encoder = nn.Sequential(
            audioConv2d(1, 32, kernel_size=3, stride=1, padding=1),
            audioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            audioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(64, 128, kernel_size=3, stride=3, padding=1),
            audioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            audioConv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(256, 512, kernel_size=3, stride=1, padding=0),
            audioConv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = paddle.reshape(x, (x.shape[0], -1))
        return x


class Generator(nn.Layer):

    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, isconcat=True, narrow=1):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.LayerList()
        self.upsamples = nn.LayerList()
        self.to_rgbs = nn.LayerList()
        self.to_outtermasks = nn.LayerList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel * self.feat_multiplier,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                ))

            self.convs.append(StyledConv(out_channel * self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel))

            self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2
        self.pre_latent = None

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        face_w=None,
        face_delta_w_depth=-1,
        latent_smooth_wt=0,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            noise = [None] * (2 * (self.log_size - 2) + 1)

        latent = styles[0].unsqueeze(1).tile((1, self.n_latent, 1))
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip
        return image, latent


class FullGenerator(nn.Layer):

    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        outter_mask=True,
        narrow=1,
        add_mask=None,
        mask_path="",
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }
        self.log_size = int(math.log(size, 2))
        self.style_dim = style_dim
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, \
            isconcat=isconcat,
            narrow=narrow)

        conv = [ConvLayer(6, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel
        self.face_final = nn.Sequential(audioConv2d(out_channel, style_dim, kernel_size=4, stride=1, padding=0),
                                        audioConv2d(style_dim, style_dim, kernel_size=1, stride=1, padding=0))
        self.audio_encoder = AudioEncoder()
        self.act = nn.Sigmoid()

        self.add_mask = add_mask
        if self.add_mask:
            mask_gt = cv2.imread(mask_path)
            mask_gt = 1. - mask_gt[:, :, 0] / 255.
            mask_gt_256 = paddle.to_tensor(mask_gt, dtype="float32").reshape((1, 1, 256, 256))
            mask_gt = cv2.resize(mask_gt, (128, 128))
            mask_gt_128 = paddle.to_tensor(mask_gt, dtype="float32").reshape((1, 1, 128, 128))
            mask_gt = cv2.resize(mask_gt, (64, 64))
            mask_gt_64 = paddle.to_tensor(mask_gt, dtype="float32").reshape((1, 1, 64, 64))
            mask_gt = cv2.resize(mask_gt, (32, 32))
            mask_gt_32 = paddle.to_tensor(mask_gt, dtype="float32").reshape((1, 1, 32, 32))
            self.mask_list = [mask_gt_32, mask_gt_64, mask_gt_128, mask_gt_256][::-1]
            [self.register_buffer('mask_%s' % i, x, persistable=True) for i, x in enumerate(self.mask_list)]

        self.emb_final_mlp = nn.Sequential(
            EqualLinear(1024, 512, activation='fused_lrelu'),
            EqualLinear(512, 512, activation='fused_lrelu'),
        )

    def forward(
        self,
        face_sequences,
        audio_sequences,
        pre_audio_feat=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        audio_smooth_wt=0.3,
        mask_depth=4,
        audio_amp_wt=1.2,
    ):

        audio = audio_sequences
        face = face_sequences

        noise = []
        ecd = None
        inputs = None
        inputs = face
        for i in range(self.log_size - 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)

        mask_list = []
        for name, buffer in self.named_buffers():
            if "mask" in name:
                mask_list.append(buffer)
        if self.add_mask:
            for j in range(mask_depth):
                noise_local = noise[j]
                mask_local = mask_list[j]
                noise[j] = noise_local * mask_local

        audio_feat = self.audio_encoder(audio)
        face_feat = self.face_final(inputs).reshape((audio_feat.shape[0], -1))
        audio_feat = audio_feat * audio_amp_wt
        if audio_smooth_wt > 0:
            pre_audio_feat = audio_feat if pre_audio_feat is None else pre_audio_feat
            audio_feat = audio_smooth_wt * pre_audio_feat + audio_feat * (1 - audio_smooth_wt)
        pre_audio_feat = audio_feat
        outs = self.emb_final_mlp(paddle.concat([audio_feat, face_feat], axis=1))

        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        outs = self.generator(
            [outs],
            return_latents,
            inject_index,
            truncation,
            truncation_latent,
            input_is_latent,
            noise=noise[1:],
        )
        image, latent = outs
        image = self.act(image)
        return image, pre_audio_feat


if __name__ == '__main__':
    model = FullGenerator(256, 512, 8)
    img_batch = paddle.ones((10, 6, 256, 256))
    mel_batch = paddle.ones((10, 1, 80, 16))
    x = model(img_batch, mel_batch)[0]
    print(x.shape)
