"""
Implements model trainer.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Dict, Tuple, Callable, Any
from warnings import warn

from utils.data_utils import batch_to_device, to_numpy
from utils.loss_functions import MTloss
from utils.data_utils import unstandardize
from utils.lr_scheduler import cos_decay_with_warmup

# Removes pandas warning.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

TASKS = ['q', 'et', 'swe', 'tws']


class Trainer(object):
    """Model trainer class.

    Parameters
    ----------
    model
        A BaseModule that implements a forward function.
    train_loader
        DataLoader that reads training batches.
    valid_loader
        DataLoader that reads training batches.
    learning rate
        The learning rate, a float > 0, passed to the optimizer.
    weight_decay
        Weight decay (L2 penalty).
    gradient_clipping (default: None)
        Gradient clipping can avoid cradient explosion by clipping the
        gradients before updating the parameters. The default ``None``
        disables gradient clipping.
    task_weighting
        If ``True``, task weighting is done: The weights correspond to
        task uncertainty that is used to balance the tasks and shift
        focus on the tasks during training.
    device
        The device to run model on (e.g. 'cuda' or 'cpu').
    optimizer_kwargs
        Additional kwargs passed to the optimizer.
    kwargs
        No effect.
    """

    def __init__(
            self,
            model,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            test_loader: DataLoader,
            all_loader: DataLoader,
            learning_rate: float,
            learning_rate_taskw: float,
            weight_decay: float,
            gradient_clipping: float,
            task_weighting: bool,
            device: str,
            num_runs_before_spinup=0,
            logdir: str = '/tmp',
            optimizer_kwargs: Dict[str, Any] = {},
            **kwargs) -> None:

        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.all_loader = all_loader
        self.learning_rate = learning_rate
        self.learning_rate_tasksw = learning_rate_taskw
        self.gradient_clipping = gradient_clipping
        self.task_weighting = task_weighting

        self.logdir = logdir
        self.imgdir = os.path.join(logdir, 'imgs')

        # Data reduction bins and slices that are used to align predictions with target.
        all_dataset_names = train_loader.dataset.features_dynamic + \
            train_loader.dataset.targets

        self.data_stats = {
            ds: {
                'mean': torch.tensor(train_loader.dataset.get_mean_std(ds)[0], device=self.device),
                'std': torch.tensor(
                    train_loader.dataset.get_mean_std(ds)[1],
                    device=self.device)
            } for ds in all_dataset_names
        }

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.mt_loss = MTloss(
            task_weighting=task_weighting,
            data_stats=self.data_stats,
            device=self.device)

        if task_weighting:
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        'params': self.model.parameters(),
                        'lr': learning_rate,
                        'weight_decay': weight_decay
                    },
                    {
                        'params': self.mt_loss.parameters(),
                        'lr': learning_rate_taskw,
                        'weight_decay': 0.0
                    }],
                **optimizer_kwargs
            )
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        'params': self.model.parameters(),
                        'lr': learning_rate,
                        'weight_decay': weight_decay
                    }
                ],
                **optimizer_kwargs
            )

        scheduler_lambda = cos_decay_with_warmup(
            warmup=0, T=120)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=scheduler_lambda)
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=1.0)  # 0.94
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1)

        self.variables = train_loader.dataset.features_dynamic + train_loader.dataset.targets

        # Data reduction bins and slices that are used to align predictions with target.
        self.slices = {
            'train': {ds: train_loader.dataset.get_slice(ds) for ds in self.variables},
            'valid': {ds: valid_loader.dataset.get_slice(ds) for ds in self.variables},
            'test': {ds: test_loader.dataset.get_slice(ds) for ds in self.variables},
            'all': {ds: all_loader.dataset.get_slice(ds) for ds in self.variables}
        }
        self.bins = {
            'train': {ds: train_loader.dataset.get_bins(ds) for ds in TASKS},
            'valid': {ds: valid_loader.dataset.get_bins(ds) for ds in TASKS},
            'test':  {ds: test_loader.dataset.get_bins(ds) for ds in TASKS},
            'all':  {ds: all_loader.dataset.get_bins(ds) for ds in TASKS}
        }

        self.units = {
            ds: train_loader.dataset.get_unit(ds) for ds in self.variables
        }

        # These are the time dimensions for all variables.
        self.ds_time = {
            ds: train_loader.dataset.getvartime(ds) for ds in self.variables
        }

        # This is the reference time dim, equal to the features time dim. All raw predictions (before
        # mapping to target resolution) are on this time intervals.
        self.ref_time = self.ds_time['tair']

        self.num_runs_before_spinup = num_runs_before_spinup

        self.epoch = 0

        # Early stopping attributes.
        self.current_best = np.inf  # Current best loss.
        # How many consecutive epochs had worse performance than current_best.
        self.patience_counter = 0

    def forward(
            self,
            feat_spinup: Dict[str, torch.Tensor],
            feat: Dict[str, torch.Tensor],
            targ: Dict[str, torch.Tensor],
            cv_set: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        batch_size = feat_spinup['tair'].size(0)

        # Initialize state variables.
        h_t = torch.zeros(
            batch_size, self.model.num_hidden, device=self.device)
        c_t = torch.zeros(
            batch_size, self.model.num_hidden, device=self.device)
        cwd_t = torch.ones(batch_size, 1, device=self.device) * -100
        gw_t = torch.ones(batch_size, 1, device=self.device) * 10
        swe_t = torch.zeros(batch_size, 1, device=self.device)

        static = feat['static'].to(self.device, non_blocking=True)
        pred_spinup = None

        # SPIN_UP
        if (self.epoch >= self.num_runs_before_spinup) or (cv_set == 'test') or ((cv_set == 'all')):
            with torch.no_grad():
                # Stack dynamic features, move seq dim to position 0.
                x = torch.cat((
                    feat_spinup['tair'],
                    feat_spinup['prec'],
                    feat_spinup['rn']), dim=-1).permute(1, 0, 2)

                # Move data to devive.
                x = x.to(self.device, non_blocking=True)

                # Get unstandardized precip and rn, move seq dim to position 0.
                prec = unstandardize(
                    x.narrow(-1, 1, 1), self.data_stats['prec'])
                rn = unstandardize(
                    x.narrow(-1, 2, 1), self.data_stats['rn'])
                tair = unstandardize(
                    x.narrow(-1, 0, 1), self.data_stats['tair'])

                # Run model.
                pred_spinup, h_t, c_t, cwd_t, gw_t, swe_t = self.model(
                    x=x,
                    prec=prec,
                    rn=rn,
                    tair=tair,
                    static=static,
                    h_t=h_t,
                    c_t=c_t,
                    cwd_t=cwd_t,
                    gw_t=gw_t,
                    swe_t=swe_t,
                    epoch=self.epoch
                )

                del feat_spinup

        # FORWARD RUN

        # Stack dynamic features, move seq dim to position 0.
        x = torch.cat((
            feat['tair'],
            feat['prec'],
            feat['rn']), dim=-1).permute(1, 0, 2)

        # Move data to devive.
        x = x.to(self.device, non_blocking=True)

        # Get unstandardized precip and rn, move seq dim to position 0.
        prec = unstandardize(
            x.narrow(-1, 1, 1), self.data_stats['prec'])
        rn = unstandardize(
            x.narrow(-1, 2, 1), self.data_stats['rn'])
        tair = unstandardize(
            x.narrow(-1, 0, 1), self.data_stats['tair'])

        targ = batch_to_device(targ, self.device)

        # Run model.
        pred, _, _, _, _, _ = self.model(
            x=x,
            prec=prec,
            rn=rn,
            tair=tair,
            static=static,
            h_t=h_t,
            c_t=c_t,
            cwd_t=cwd_t,
            gw_t=gw_t,
            swe_t=swe_t,
            epoch=self.epoch
        )

        self.bin_reduce_(
            x=pred,
            sample_set=cv_set)

        loss = self.mt_loss(
            pred=pred,
            target=targ,
            cv_set=cv_set)

        return loss, pred, pred_spinup

    def train_epoch(self, num_train_batches=None) -> None:
        self.model.train()
        self.epoch += 1

        n_iter = num_train_batches if num_train_batches else len(
            self.train_loader)

        nan_counter = 0

        for step, (feat_spinup, feat, targ, loc) in enumerate(self.train_loader):

            # print(f'training: {step + 1:3d} of {n_iter}')

            loss, pred, _ = self.forward(
                feat_spinup=feat_spinup,
                feat=feat,
                targ=targ,
                cv_set='train')

            if torch.isnan(loss):
                # This is a debugging feature, if NaNs occure, possible a bug and not
                # just training instability.
                nan_counter += 1
                if nan_counter > 9:
                    raise ValueError(
                        'Training loss was NaN >5 times, training stopped.')
                warn(
                    f'Training loss was NaN {nan_counter} time{"" if nan_counter==1 else "s"} '
                    'in a row, will stop after >9.')

                continue

            # self.scheduler.step(self.epoch + step / n_iter)
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            del loss, pred, feat, targ, loc

            if step >= n_iter:
                break

        self.scheduler.step()
        metrics_sum = self.mt_loss.get_epoch_summary()
        metrics_sum.update({'epoch': self.epoch})
        metrics_sum = self.add_lr_metrics(metrics=metrics_sum)

        return metrics_sum

    @torch.no_grad()
    def eval_epoch(self, num_train_batches=None) -> None:
        self.model.eval()

        n_iter = num_train_batches if num_train_batches else len(
            self.valid_loader)

        for step, (feat_spinup, feat, targ, loc) in enumerate(self.valid_loader):

            # print(f'evaluation: {step + 1:3d} of {n_iter}')

            loss, pred, _ = self.forward(
                feat_spinup=feat_spinup,
                feat=feat,
                targ=targ,
                cv_set='valid'
            )

            if torch.isnan(loss):
                raise ValueError('Loss is NaN (validation), training stopped.')

            del loss, pred, feat, targ, loc

            if step >= n_iter:
                break

        metrics_sum = self.mt_loss.get_epoch_summary()
        metrics_sum.update({'epoch': self.epoch})

        self.early_stopping(loss=metrics_sum['uloss_valid'])
        metrics_sum.update({'patience_counter': self.patience_counter})

        return metrics_sum

    @torch.no_grad()
    def predict(self, predict_all=False) -> None:
        self.model.eval()
        self.model.predict()

        result = []
        result_spinup = []

        if predict_all:
            loader = self.all_loader
        else:
            loader = self.test_loader
            loader.dataset.space_indices['test'] = np.concatenate((
                loader.dataset.space_indices['train'],
                loader.dataset.space_indices['valid'],
                loader.dataset.space_indices['test']
            ), axis=0)

        for step, (feat_spinup, feat, targ, loc) in enumerate(loader):

            print(f'prediction step {step+1:3d} / {len(loader)}')

            loss, pred, pred_spinup = self.forward(
                feat_spinup=feat_spinup,
                feat=feat,
                targ=targ,
                cv_set='all' if predict_all else 'test'
            )

            pred.update({
                'lat': loc[0],
                'lon': loc[1],
                'static_enc': pred['stat_enc']
            })
            result.append(to_numpy(pred))

            if pred_spinup:
                pred_spinup.update({
                    'lat': loc[0],
                    'lon': loc[1],
                    'static_enc': pred_spinup['stat_enc']
                })
                result_spinup.append(to_numpy(pred_spinup))

            del pred, pred_spinup, feat, targ, loc

        metrics_sum = self.mt_loss.get_epoch_summary()
        metrics_sum.update({'epoch': self.epoch})

        return metrics_sum, result, result_spinup

    def early_stopping(self, loss: float) -> None:
        """Counts how many consecutive losses are worse than current best.
        """

        if loss < self.current_best:
            self.current_best = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def save(self, checkpoint: str) -> None:
        """Saves the model at the provided checkpoint.

        Parameters
        ----------
        checkpoint
            Path to target checkpoint file.
¨
        Returns
        ----------
        checkpoint

        """
        savefile = os.path.join(checkpoint, 'chkp.pt')
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'mt_loss_state_dict': self.mt_loss.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            savefile
        )
        return savefile

    def restore(self, checkpoint: str) -> None:
        """Restores the model from a provided checkpoint.

        Parameters
        ----------
        filename
            Path to checkpoint file.

        """
        checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to_device(self.device)

        self.mt_loss.load_state_dict(checkpoint['mt_loss_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']

    def bin_reduce_(
            self,
            x: Dict[str, torch.Tensor],
            sample_set: str,
            fun: Callable = torch.mean) -> None:
        """Inplace reduction of tensors time dimension (dim 1).

        Parameters
        ----------
        x
            A dict of tensors of which all items with key in ``keys`` get reduced
            in time dimension to match target resolution. The reduced items are added to
            the passed dicts with ``key`` as key prefix. The tensors must be of shape
            (batch_size, sequence_length, ...).
        sample_set
            Sample set, one of {'train', 'valid', 'test'}.
        fun
            A function used to reduce the bins , default is ``torch.mean``.

        Returns
        ----------
        reduced: Dict[Dict[torch.Tensor]]
            Same format as input ``x`` but tensors with shape
            (batch_size, reduced_sequence_length, ...).

        """

        if sample_set not in ['train', 'valid', 'test', 'all']:
            raise ValueError(
                'Argument ``sample_set`` must be one of {"train", "valid", "test"} but is {}.'.format(sample_set))

        data_bins = self.bins[sample_set]

        for key in TASKS:

            try:
                bins = data_bins[key]
            except Exception:
                raise ValueError('Key ``{}`` not found in ``self.bins``. Valid keys are: {}.'.format(
                    key, list(data_bins.keys())))

            # If bins is a list of two values, it contains lower / upper range and data is
            # already on target resolution.
            if np.ndim(bins) == 1:
                x.update({
                    key + '_agg':
                         x[key][:, bins[0]:bins[1], ...]
                         })
            # If bins is a list of lists of two values, each of these lists marks lower /
            # upper range of a bin that is reduced to one values corresponding to target.
            else:
                x.update({
                    key + '_agg':
                    torch.stack(
                        [fun(x[key][:, b[0]:b[1], ...], dim=1)
                         for b in bins],
                        dim=1
                    )
                })

    def add_lr_metrics(
            self,
            metrics: Dict[str, Any]) -> None:
        """Add learning rate log metrics.

        Parameters
        ----------
        metrics
            Metrics to logg
        lr
            The current learning rate.
        """

        lr = self.scheduler.get_last_lr()

        # If lr is one element, it is the standard lr. If it contains two elements,
        # it is a separated lr for the task weights in addition.
        if lr is not None:
            if len(lr) > 2:
                raise ValueError('More than 2 learning rates in scheduler.')

            metrics.update({f'lr': lr[0]})
            if len(lr) == 2:
                metrics.update({f'weight_lr': lr[1]})

        return metrics

    def __repr__(self) -> str:
        s = 'TaskScheduler\n{}\n  Tasks: {}\n\nConnected {}'.format(
            ''.join(['-'] * 42),
            TASKS,
            self.bucket.__repr__())
        return s
