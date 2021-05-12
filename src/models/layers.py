
from torch.nn.utils import weight_norm
import torch.nn as nn
import torch

from typing import Optional, Iterable, Tuple, Union
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MultiLinear(nn.Module):
    """Defines multiple layers of linear layers.

    The multilayer network is a stack of ``num_layers`` linear layers, each
    of kind:
    - linear
    - dropout
    - activation

    The module subclasses Pytorch's torch.nn.Module.

    Shapes
    ----------
    - Input:  (..., input_size)
    - Output: (..., output_size)

    Parameters
    ----------
    input_size: int
        Size of input samples.
    hidden_size: int
        Number of nodes in linear layers, only applies if ``num_layers``>1.
    output_size: int
        Size of each output sample.
    num_layers: int
        Number of linear layers. Setting this to 0 will create an identity
        model returning the input without any interaction. In this case,
        you need still to pass a value for ``hidden_size``, even though
        it does not have any effect.
    dropout: float (default: 0.0)
        Dropout probability (0-1) applies after earhc linear layer.
    activation: torch.nn.modules.activation (default: nn.Sigmoid())
        Activation function being applies after each linear layer.
    dropout_last: bool (default: True)
        If ``True`` (default), a dropout layer is added after the last
        layer, else not.
    activation_last: bool (default: True)
        If ``True`` (default), an activation layer is added after the last
        layer, else not.
    *args, **kwargs: Optional
        Passed to each torch.nn.Linear layer.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            dropout: float = 0.0,
            activation: torch.nn.modules.activation = nn.LeakyReLU(),
            dropout_last: bool = False,
            activation_last: bool = False,
            *args: Optional,
            **kwargs: Optional) -> None:

        super(MultiLinear, self).__init__()
        if num_layers >= 1:
            sizes_in = [
                *[input_size],
                *[hidden_size] * (num_layers - 1)  #  Only applies if num_layers>1.
            ]

            # Output sizes of the shared layer.
            sizes_out = [
                *[hidden_size] * (num_layers - 1),
                #  Only applies if num_layers>1.
                *[output_size]
            ]

            layers = []
            for l, (i, o) in enumerate(zip(sizes_in, sizes_out)):
                layers.append(
                    nn.Linear(in_features=i, out_features=o, *args, **kwargs))

                # For last layer:
                if (l + 1) == num_layers:
                    if dropout_last:
                        layers.append(nn.Dropout(p=dropout))

                    if activation_last:
                        layers.append(activation)

                else:
                    layers.append(nn.Dropout(p=dropout))
                    layers.append(activation)

            self.model = nn.Sequential(*layers)

        else:
            #  Identity model.
            self.model = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiCNN(nn.Module):
    """Defines multiple layers of CNNs.

    The multilayer network is a stack of ``num_layers`` CNNs, each
    of kind:
    - Conv2D
    - ReLU
    - MaxPool2d

    ...followed by a number of linear layers.

    The module subclasses Pytorch's torch.nn.Module.

    This module does not allow to have non-symmetric kernels or image sizes.

    Shapes
    ----------
    - Input:  (..., input_size, H, W)

    Parameters
    ----------
    in_channels
        Number of channels in the input
    out_channels
        Number of channels produced by the convolution
    kernel_size
        Size of the convolving kernel
    stride (default: 1)
        Stride of the convolution.
    padding (default: 0)
        Zero-padding added to both sides of the input.
    dilation (default: 1)
        Spacing between kernel elements.
    in_size (default: 30)
        The input image width and height.
    linear_hidden_size (default: 32)
        The linear layer hidden layer sizes.
    linear_output_size (default: 1)
        The linear layer final output size.
    linear_num_layers (default: 1)
        The number of linear layers.
    dropout: (default: 0.0)
        Dropout probability applied after last CNN layer and after each
        hidden linear layer.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: Union[int, Tuple[int]],
            kernel_size: Union[int, Tuple[int]],
            pool_kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            dilation: Union[int, Tuple[int]] = 1,
            in_size: int = 30,
            linear_hidden_size: int = 32,
            linear_output_size: int = 1,
            linear_num_layers: int = 1,
            dropout: float = 0.0) -> None:

        super(MultiCNN, self).__init__()

        if not isinstance(out_channels, tuple):
            out_channels = (out_channels,)

        self.num_layers = len(out_channels)

        kernel_size = self.toTuple(kernel_size, self.num_layers)
        pool_kernel_size = self.toTuple(pool_kernel_size, self.num_layers)
        stride = self.toTuple(stride, self.num_layers)
        padding = self.toTuple(padding, self.num_layers)
        dilation = self.toTuple(dilation, self.num_layers)

        layers = []
        in_channels_i = in_channels
        in_size_i = in_size
        for i in range(self.num_layers):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels_i,
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i]),
            )
            layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=pool_kernel_size, stride=pool_kernel_size)
            )
            layers.append(
                torch.nn.ReLU()
            )
            layers.append(
                nn.BatchNorm2d(out_channels[i])
            )

            in_channels_i = out_channels[i]
            in_size_i = self.inferSize(
                in_size_i, kernel_size[i], padding[i], dilation[i], stride[i])
            in_size_i = self.inferSize(
                in_size_i, pool_kernel_size[i], stride=pool_kernel_size[i])

        if in_size_i <= 0:
            raise ValueError(
                f'The output size of the last convolutional layer ({in_size_i}) is '
                '≤ 0, which is not valid. Change arguments.')

        layers.append(torch.nn.Dropout(dropout))

        layers.append(Flatten())

        layers.append(MultiLinear(
            input_size=in_channels_i * in_size_i ** 2,
            hidden_size=linear_hidden_size,
            output_size=linear_output_size,
            num_layers=linear_num_layers,
            dropout=dropout)
        )

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def toTuple(self, x, num_layers=1) -> Tuple[int]:
        if not isinstance(x, tuple):
            if x is not None:
                x = (x,) * num_layers
        else:
            if len(x) != num_layers:
                raise ValueError(
                    f'The number of layers ({self.num_layers}) is infered from the length of '
                    '`out_channels`. Each of the following arguments must either '
                    'be an integer (in this case, it is applied to each layer) or a tuple '
                    f'of length `num_layers` ({self.num_layers}):'
                    '\n- `kernel_size`'
                    '\n- `stride`'
                    '\n- `padding`'
                    '\n- `dilation`')
        return x

    def inferSize(self, size, kernel_size, padding=0, dilation=1, stride=1):
        """Infer output size of a convolutional layer

        https://pytorch.org/docs/stable/nn.html#conv2d

        """

        out_size = np.floor(
            (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

        return int(out_size)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2) -> None:

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1,
                                 self.conv2, self.chomp2, self.relu2,
                                 self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
