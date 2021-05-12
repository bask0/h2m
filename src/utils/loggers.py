import torch
import pprint


class EpochLogger(object):
    """Keeps track of values sum and count.

    You can add values to the logger and if they are already regisered,
    the sum as well as the count will be recorded. The mean value can
    be retrieved and by default, this will reset the values.

    Usage
    ----------
    epoch_logger = EpochLogger()

    for batch in batches:
        ...
        epoch_logger.log('loss', 'training', loss)

    # Get summary and reset logger for next epoch.
    sumary = epoch_logger.get_summary()

    """

    def __init__(self):
        self.summary = {}

    def log(self, name, cv_set, value):
        """Log variable.

        In the first iteration of an epoch, the key is created in the summary, in
        all further iterations, the value is added to an existing key.

        Parameters
        ----------
        name:   The name of the value to log, a string
        cv_set: The dataset, something like 'train' or 'test', as string that
                will be added to 'name'
        value:  A single value to log, can be numeric, torch.Tensor allowed.
        """

        name = f'{name}_{cv_set}'
        if name in self.summary:
            self.summary[name]['val'] += value
            self.summary[name]['n'] += 1
        else:
            self.summary[name] = {
                'val': value,
                'n': 1
            }

    def reset(self):
        """Reset the epoch logger.

        This should be done at the end of each epoch, but is the default behavior
        when calling 'get_summary'.
        """
        self.summary = {}

    def get_summary(self, reset=True):
        """Get epoch summary.

        The mean of the logged values over the epoch are returned.

        Parameters
        ----------
        reset:      If True (default), the logger will be reset

        Returns
        ----------
        A dict of mean values over the epoch.
        """
        summary = {}
        for k, v in self.summary.items():
            summary.update({
                k: self.to_numeric(v['val']) / v['n']
            })

        if reset:
            self.reset()

        return summary

    def to_numeric(self, x):
        """Convert torch.Tensors to numeric."""
        if isinstance(x, torch.Tensor):
            return x.item()
        else:
            return x

    def __repr__(self):
        return f'EpochLogger:\nTracked values:\n{self.summary}'

    def __str__(self):
        return f'EpochLogger:\nTracked values:\n{pprint.pformat(self.summary)}'
