import torch
from torch import jit
from utils.loggers import EpochLogger
from utils.data_utils import standardize
import os

from torch import Tensor


@jit.script
def minimize_positives(x: Tensor, bias: float = 0.1) -> Tensor:
    """Mean of all positive values (push positives towards 0).

    To prevent the task weight from converging to -inf (as a loss of
    0 is plausible), a bias is added to the loss. This value should
    be in the range of the other losses.

    The loss is first calculated per batch and then averaged over
    batches.

    Parameters
    ----------
    x: torch.Tensor
        Tensor to calculate the constraint loss on, with
        shape (batch, ...).
    bias: float
        A scalar that is added to the loss.

    Returns
    ----------
    loss: torch.Tensor
        The constraint loss, a tensor of one element.
    """

    red_dims = [d for d in range(1, x.dim())]
    return torch.relu(x).mean(red_dims).mean() + bias


@jit.script
def positive_constraint(x: Tensor, bias: float = 0.1) -> Tensor:
    """Mean of all negative values (push negatives towards 0).

    To prevent the task weight from converging to -inf (as a loss of
    0 is plausible), a bias is added to the loss. This value should
    be in the range of the other losses.

    The loss is first calculated per batch and then averaged over
    batches.

    Parameters
    ----------
    x: torch.Tensor
        Tensor to calculate the constraint loss on, with
        shape (batch, ...).
    bias: float
        A scalar that is added to the loss.

    Returns
    ----------
    loss: torch.Tensor
        The constraint loss, a tensor of one element.
    """

    red_dims = [d for d in range(1, x.dim())]
    return torch.relu(-x).mean(red_dims).mean() + bias


@jit.script
def top_quantile_positive_constraint(x: Tensor, q: float, bias: float = 0.1) -> Tensor:
    """Mean of all negative values in the top q quantile.

    To prevent the task weight from converging to -inf (as a loss of
    0 is plausible), a bias is added to the loss. This value should
    be in the range of the other losses.

    The loss is first calculated per batch and then averaged over
    batches.

    Parameters
    ----------
    x: torch.Tensor
        2D tensor to calculate the constraint loss on, with
        shape (batch, seq).
    q: float
        Top quantile (0, 1) that must be above 0. E.g. with p=0.1,
        10 % of x along the last dimensions must be lager than 0.
    bias: float
        A scalar that is added to the loss.

    Returns
    ----------
    loss: torch.Tensor
        The constraint loss, a tensor of one element.
    """

    if x.dim() != 2:
        raise ValueError(
            f'operation only supported for 2D tensors (got {x.dim()}D).'
        )

    if not (0. <= q <= 1.):
        raise ValueError(
            f'`q` must be in range [0, 1] but is {q}.'
        )

    if q == 0.:
        return torch.relu(-x.max(dim=1)[0]).mean() + bias
    if q == 1.:
        return positive_constraint(x, bias=bias)
    else:
        k = int(q * x.size(1))
        return torch.relu(-x.topk(k, dim=1)[0]).mean() + bias


@jit.script
def abs_mean(x, bias: float = 0.1) -> Tensor:
    """Mean fo all absolute values.

    To prevent the task weight from converging to -inf (as a loss of
    0 is plausible), a bias is added to the loss. This value should
    be in the range of the other losses.

    Parameters
    ----------
    x: torch.Tensor
        The tensor to scalculate the constraint loss on.

    Returns
    ----------
    loss: torch.Tensor
        The constraint loss, a tensor of one element.
    bias: float
        A scalar that is added to the loss.
    """

    return torch.abs(x).mean() + bias


@jit.script
def nanmae(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """Mean absolute error (MAE) loss, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The mae loss, a tensor of one element.
    """

    cnt = torch.any(torch.isfinite(target), dim=-1).sum(dtype=target.dtype)
    mae = torch.nansum(torch.abs(pred-target), dim=-1)

    return mae.sum() / cnt


@jit.script
def nanmse(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """Mean squared error (MSE) loss, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The mse loss, a tensor of one element.
    """

    mask = torch.isnan(target)
    cnt = torch.sum(~mask, dtype=target.dtype)

    mse = torch.pow(pred - target, 2).sum() / cnt

    return mse


@jit.script
def nanrmse(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """root mean squared error (RMSE) loss, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The rmse loss, a tensor of one element.
    """

    return torch.sqrt(nanmse(pred, target))


@jit.script
def nannse(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """Nash-Sutcliffe modeling efficiency, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The nse loss, a tensor of one element.
    """
    pred = pred.squeeze()
    target = target.squeeze()

    all_nan = torch.all(torch.isnan(target), dim=-1)
    pred[all_nan] = 0.0
    target[all_nan] = 0.0

    mask = torch.isnan(target)

    a = torch.sum((pred - torch.where(mask, pred, target)) ** 2, dim=-1, keepdim=True)

    t_mean = torch.nansum(target, dim=-1, keepdim=True) / \
        torch.sum(~mask, dim=-1, dtype=target.dtype, keepdim=True)
    b = torch.sum((torch.where(mask, t_mean, target) - t_mean) ** 2, dim=-1)

    a.clamp_(min=0.0001)
    b.clamp_(min=0.0001)

    return torch.mean(1. - a / b)


@jit.script
def nannse_norm(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """Adapted* Nash-Sutcliffe modeling efficiency, handels NaN values.

    The NSE is normalized to range 0-1 and flipped: A value of 1 is the highest
    possible loss, and a values of 0 the lowest (best). A value of 0.5 indicates that
    the performance is equivalent to taking the mean value as prediction.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The nse loss, a tensor of one element.
    """
    return 1 - nannse(pred, target)


def hook_fn(m, i, o):
    m.step += 1


class MTloss(torch.nn.Module):
    """Multi-task loss module.


    Parameters
    ----------
    task_weighting: bool
        If true, automatic task weighting is done.
    data_stats: Dict[task: {mean: ..., std: ...}]
        Data statistics for the target variables.
    device: str
        Device, cpu or cuda.
    debug_tasks: List[str]
        A list of task names that are exclusively added to the loss function. E.g. if ['et'],
        only 'et' will be optimized.
    """
    def __init__(
            self,
            task_weighting=False,
            data_stats=None,
            device='cuda',
            debug_tasks=None):
        super(MTloss, self).__init__()

        # For compatibility.
        if data_stats is None:
            raise ValueError('Argument `data_stats` cannot be `None`.')

        self.logger = EpochLogger()
        self.data_stats = data_stats
        self.task_weighting = task_weighting
        self.debug_tasks = debug_tasks

        self.loss_sigmas = torch.nn.ParameterDict({
            task: torch.nn.Parameter(
                torch.ones(1, device=device) * val,
                requires_grad=task_weighting) for val, task in zip(
                    [0.7, 0.7, 0.7, 0.7, 0.7, 0.7], ['tws', 'et', 'swe', 'q', 'cwd', 'snow'])  # 0.5 / x ** 2 -> w
        })

        self.swe_saturation_threshold = 120.0

        # Step is incremented in each backward call to the module.
        self.step = torch.zeros(1, device=device)
        self.register_backward_hook(hook_fn)

    def get_epoch_summary(self):
        return self.logger.get_summary()

    def forward(self, pred, target, cv_set):

        DEBUG = os.environ['debug'] == 'True'

        total_uloss = 0.0
        total_wloss = 0.0

        for task in ['tws', 'et', 'swe', 'q']:

            if DEBUG:
                print('TASK: ', task)

            P = pred[task + '_agg']
            T = target[task]

            if task == 'swe':
                P = P.clamp(None, self.swe_saturation_threshold)
                T = T.clamp(None, self.swe_saturation_threshold)

            P = standardize(
                P, self.data_stats[task])
            T = standardize(
                T, self.data_stats[task])

            if task == 'tws':
                P = P - P.mean(dim=(1, 2), keepdim=True)
                T = T - T.mean(dim=(1, 2), keepdim=True)

            uloss = nanmae(P, T)

            s = torch.log(self.loss_sigmas[task] ** 2)  # log sigma2
            w = 0.5 * torch.exp(-s)  # precision: 1 / sigma2
            r = s * 0.5  # log sigma
            wloss = uloss * w + r

            if self.debug_tasks is None or task in self.debug_tasks:
                if DEBUG:
                    print(f'  {task}: added loss to total loss')
                total_uloss += uloss
                total_wloss += wloss[0]

            if DEBUG:
                print('  uloss: ', uloss)
                print('  s    : ', s)
                print('  w    : ', w)
                print('  wloss: ', wloss)

            self.logger.log(task + '_uloss', cv_set, uloss.item())
            self.logger.log(task + '_wloss', cv_set, wloss.item())
            self.logger.log(task + '_weight', cv_set, w.item())

            if DEBUG:
                print(self.logger)

        for task in ['cwd']:
            if task == 'cwd':
                # The weight is decreased over time.
                mixing = 1 - torch.sigmoid((self.step - 30) / 4)
                # Pushes the top 1% of cwd towards zero. Note that bias is added after mixing.
                uloss = top_quantile_positive_constraint(
                    pred[task].squeeze(), q=0.1, bias=0.0) * mixing + 0.1
                uloss = uloss[0]
            elif task == 'snow':
                uloss = minimize_positives(
                    pred['smelt'] * pred['sfrac'])
            else:
                uloss = positive_constraint(pred[task])
            s = torch.log(self.loss_sigmas[task] ** 2)  # log sigma2
            w = 0.5 * torch.exp(-s)  # precision: 1 / sigma2
            r = s * 0.5  # log sigma
            wloss = uloss * w + r

            if self.debug_tasks is None or task in self.debug_tasks:
                if DEBUG:
                    print(f'  {task}: added loss to total loss')
                total_uloss += uloss
                total_wloss += wloss[0]
            self.logger.log(task + '_constr_uloss', cv_set, uloss.item())
            self.logger.log(task + '_constr_wloss', cv_set, wloss.item())
            self.logger.log(task + '_constr_weight', cv_set, w.item())

        self.logger.log('uloss', cv_set, total_uloss.item())
        self.logger.log('wloss', cv_set, total_wloss.item())

        if self.task_weighting:
            self.logger.log('loss', cv_set, total_wloss.item())
            return total_wloss
        else:
            self.logger.log('loss', cv_set, total_uloss.item())
            return total_uloss
