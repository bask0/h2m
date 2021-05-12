import ConfigSpace as CS
from typing import Dict


def get_config(experiment: str, name: str) -> Dict:
    """Get a model configuration.

    Parameters
    ----------
    experiment
        Experiment name.
    name
        Configuration name.

    Returns
    ----------
    A tuple of dicts
        (hardcoded configurration, hyperparameter search space)

    """

    config_space = CS.ConfigurationSpace()

    if experiment == 'hybrid':

        # Model args.
        config_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('stat_nh', lower=50, upper=150, q=50),
            CS.UniformIntegerHyperparameter('stat_nl', lower=1, upper=2),
            CS.UniformFloatHyperparameter('stat_dropout', lower=0.0, upper=0.5, q=0.1),
            CS.UniformIntegerHyperparameter('temp_nh', lower=50, upper=100, q=50),
            CS.UniformIntegerHyperparameter('task_nh', lower=50, upper=200, q=50),
            CS.UniformIntegerHyperparameter('task_nl', lower=1, upper=2),
            CS.UniformFloatHyperparameter('task_dropout', lower=0.0, upper=0.5, q=0.1),
        ])

        # Data args.
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter('fold', choices=[0, 1, 2, 3, 4]),
            CS.CategoricalHyperparameter('offset', choices=[(0, 0)])
        ])

        # Optim args.
        config_space.add_hyperparameters([
            CS.CategoricalHyperparameter('lr', choices=[1e-1, 1e-2, 1e-3]),
            CS.CategoricalHyperparameter('w_decay', choices=[1e-2, 1e-3, 1e-4]),
            CS.UniformFloatHyperparameter('grad_clip', lower=0.001, upper=0.1, q=0.001)
        ])

        hc_config = get_hc_config()

        task_name = name.split('_')[0]

        if name == 'all_vars_no_task_weighting' or name == 'test':
            hc_config.update({'tasks': ['swe', 'et', 'q', 'tws']})
            hc_config.update({'task_weighting': False})
        elif name == 'all_vars_task_weighting' or name == 'test':
            hc_config.update({'tasks': ['swe', 'et', 'q', 'tws']})
            hc_config.update({'task_weighting': True})
            config_space.add_hyperparameters([
                CS.CategoricalHyperparameter('lr_taskw', choices=[1e-2, 1e-3])
            ])
        elif name == f'{task_name}_no_task_weighting':
            hc_config.update({'tasks': [task_name]})
            hc_config.update({'task_weighting': False})
        else:
            raise ValueError(
                f'No configuration found for experiment `{experiment}`, name `{name}`.')
        return hc_config, config_space

    else:
        raise ValueError(
            f'No configuration found for experiment `{experiment}`.')


def get_hc_config() -> Dict:
    """Hard-coded configuration (not part of hyperparameter search)."""

    data_args = dict(
        dataset='/scratch/hydrodl/data/bucket.zarr',
        dataconfig='/workspace/hydrodl/src/data_config.json',
        batch_size=80,
        warmup_steps='1Y',
        num_spinup_years=5,
        num_stat_enc=12,
        num_workers=6,
        seed=13,
        pin_memory=False
    )

    raytunes_args = dict(
        ncpu=10,
        ngpu=1
    )

    logging_args = dict(
        store='/scratch/hydrodl/experiments',
        overwrite=True,
        num_train_batches=None
    )

    # Early stopping used for  validation / test set.
    early_stopping_args = dict(
        patience=20,  # How many times validation loss can get worse before stopping.
        grace_period=20  # Number of epochs to wait before applying p2atience.
    )

    hpband_args = dict(
        max_t=150,
        reduction_factor=3,
        num_samples=120,  # https://github.com/ray-project/ray/issues/5775
        metric='uloss_valid',
        mode='min'
    )

    config = {
        **data_args,
        **raytunes_args,
        **logging_args,
        **early_stopping_args,
        **hpband_args
    }

    return config
