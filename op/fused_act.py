import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class FusedLeakyReLU(nn.Layer):

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        if bias:
            self.bias = self.create_parameter((channel,), default_initializer=nn.initializer.Constant(0.0))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        return (F.leaky_relu(input + bias.reshape([1, bias.shape[0]] + [1] * (len(input.shape) - 2)), negative_slope=0.2) * scale)

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale