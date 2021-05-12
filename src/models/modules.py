
import torch
from torch import nn
from torch.nn import init
import numpy as np


class BaseModule(nn.Module):
    """
    Basic Pytorch Module, meant to be subclassed.

    To create a subclass, you need to implement the following functions:
    __init__: Initialize the class; first call
        super(<ClassName>, self).__init__().
    forward: Model forward pass..

    Special methods
    ----------
    - weight_init: Call on model to initialize weights.
    - to_device: Call on model to move model to specified device.

    Some code snippets copied from
        https://github.com/vincentherrmann/pytorch-wavenet/
    """

    def __init__(self) -> None:
        super(BaseModule, self).__init__()

        # Default device.
        self.device = 'cpu'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines tha forward pass.

        Do not call this directly, use model(...) to perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            A tensor to forward passed through the model.

        Returns
        ----------
        output: torch.Tensor
            Model output.
        """

        raise NotImplementedError(
            'Method ``model`` needs to be overridden in subclass.')

    def parameter_count(self) -> int:
        """Count the number of model parameters.

        Returns
        ----------
        num_parameters: int
            The number of model parameters.
        """
        par = list(self.parameters())
        num_param = sum([np.prod(list(d.size())) for d in par])
        return num_param

    def to_device(self, device: str) -> None:
        """Move model to given device.

        Parameters
        ----------
        device: str
            The device to move model to, i.e. 'cpu' or 'cuda'.
        """
        # Remember device.
        self.device = device
        # Move model to device.
        self.to(device)

    def weight_init(self) -> None:
        """Initialize model weights with custom distributions.
        """
        self.apply(self._weight_init)

    def _weight_init(self, m) -> None:
        """Initialize model weights with custom distributions.

        Do not use this method but call ``weight_init``.
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)


class BaseModule_jit(torch.jit.ScriptModule):
    """
    Basic Pytorch Module, meant to be subclassed.

    To create a subclass, you need to implement the following functions:
    __init__: Initialize the class; first call
        super(<ClassName>, self).__init__().
    forward: Model forward pass..

    Special methods
    ----------
    - weight_init: Call on model to initialize weights.
    - to_device: Call on model to move model to specified device.

    Some code snippets copied from
        https://github.com/vincentherrmann/pytorch-wavenet/
    """

    def __init__(self) -> None:
        super(BaseModule, self).__init__()

        # Default device.
        self.device = 'cpu'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines tha forward pass.

        Do not call this directly, use model(...) to perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            A tensor to forward passed through the model.

        Returns
        ----------
        output: torch.Tensor
            Model output.
        """

        raise NotImplementedError(
            'Method ``model`` needs to be overridden in subclass.')

    def parameter_count(self) -> int:
        """Count the number of model parameters.

        Returns
        ----------
        num_parameters: int
            The number of model parameters.
        """
        par = list(self.parameters())
        num_param = sum([np.prod(list(d.size())) for d in par])
        return num_param

    def to_device(self, device: str) -> None:
        """Move model to given device.

        Parameters
        ----------
        device: str
            The device to move model to, i.e. 'cpu' or 'cuda'.
        """
        # Remember device.
        self.device = device
        # Move model to device.
        self.to(device)

    def weight_init(self) -> None:
        """Initialize model weights with custom distributions.
        """
        self.apply(self._weight_init)

    def _weight_init(self, m) -> None:
        """Initialize model weights with custom distributions.

        Do not use this method but call ``weight_init``.
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
