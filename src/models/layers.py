
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
