import os
import shutil
import glob2
import pickle
import json
from ray.tune.analysis import Analysis


class ExpManager(object):
    """Manage experiment configuration and paths."""

    @classmethod
    def load_from_path(cls, path):
        """Load ExpManager from file path.

        Parameters
        ----------
        path: str
            File path to a ExpManager .pkl file.
        """
        with open(path, 'rb') as f:
            expmanager = pickle.load(f)

        return expmanager

    @classmethod
    def load_from_args(cls, store, experiment, name, mode):
        """Load ExpManager from arguments.

        Parameters
        ----------
        store: str
            Base path.
        experiment: str
            Experiment name.
        name: str
            Run name.
        mode: str
            Mode, on of `tune` or `cv`.
        """

        _, _, _, _, expconfig_file = \
            cls.get_paths(store, experiment, name, mode)
        return cls.load_from_path(expconfig_file)

    @classmethod
    def get_paths(cls, store, experiment, name, mode):
        """Create essential paths (needed for object init and restoring).

        Parameters
        ----------
        store: str
            Base path.
        experiment: str
            Experiment name.
        name: str
            Run name.
        mode: str
            Mode, on of `tune` or `cv`.

        Returns
        -------
        Tuple: run_dir, tune_dir, expconfig_file
        """

        run_dir = \
            f'{store}/{experiment}/{name}/{mode}'
        tune_dir = \
            f'{store}/{experiment}/{name}/tune'

        expconfig_file = f'{run_dir}/experimentconfig.pkl'

        return run_dir, tune_dir, expconfig_file

    def __init__(self, store, experiment, name, mode, fold=None, offset=None, overwrite=False):
        """Create a PathManager to derive experiment paths.

        If `fold` and `offset` are not passed, basic experiment paths
        are derived only for tuning and cross-validation:
        - self.run_dir: the experiment path
        - self.tune_dir: the tune path (same as above if `mode` is `tune`)
        - self.summary_dir: summary directory, always refers to `tune` path
        - self.expconfig_file: experiment configuration (NOT model HPs).
        - self.best_params: the best parameters of `tune`, a .pkl file.

        If `fold` and `offset` are passed (or set later via
        `PathManager.set_fold_paths`), further paths are generated:
        - self.predictions_path: prediction directory
        - self.predictions_spinup_path: prediction directory for spinup periode

        Parameters
        ----------
        store: str
            Base path.
        experiment: str
            Experiment name.
        name: str
            Run name.
        mode: str
            Mode, on of `tune` or `cv`.
        fold: int
            Fold number. If passed, also `offset` is required.
        offset: tuple of ints
            A two element tuple with the offsets. If passed, also
            `fold` is required.
        overwrite: bool
            If `True`, the directories get overwritten. `False` is default.
        """
        self.store = store
        self.experiment = experiment
        self.name = name
        self.mode = mode
        self.overwrite = overwrite

        self.has_fold = False
        self.has_config = False

        self.tune = self.mode == 'tune'

        self.run_dir, self.tune_dir, self.expconfig_file = self.get_paths(
                self.store, self.experiment, self.name, self.mode)

        self.summary_dir = f'{self.tune_dir}/summary'
        self.summary_save_dir = f'{self.run_dir}/summary'
        self.config_file = f'{self.summary_dir}/hp_config.json'
        self.best_hp_config_file = f'{self.summary_dir}/best_params.pkl'
        self.searchspace_file = f'{self.summary_dir}/search_space.txt'

        if (fold is None) ^ (offset is None):
            raise ValueError(
                'either pass none or both of `fold` and `offset`.'
            )

        if fold is not None:
            self.set_fold_paths(fold, offset)
            self.has_fold = True

        if self.overwrite:
            shutil.rmtree(self.run_dir, ignore_errors=True)
        os.makedirs(self.summary_dir, exist_ok=True)

    def clear_predictions(self):
        """Delete prediction and spinup predictions."""
        if not self.has_fold:
            raise AssertionError(
                'Cannot clear predictions because `fold` and `offset` '
                'are needed to determine predictions directory, but '
                'they have not yet been set. Use `.set_fold_paths` to '
                'do so.'
            )
        shutil.rmtree(self.predictions_dir, ignore_errors=True)
        shutil.rmtree(self.predictions_spinup_dir, ignore_errors=True)
        shutil.rmtree(self.predictions_all_dir, ignore_errors=True)
        os.makedirs(self.predictions_dir)
        os.makedirs(self.predictions_spinup_dir)
        os.makedirs(self.predictions_all_dir)

    def set_fold_paths(self, fold, offset):
        """Set `fold` and `offset`.

        Parameters
        ----------
        fold: int
            Fold number.
        offset: tuple of ints
            A two element tuple with the offsets.
        """
        self.fold = fold
        self.offset = offset

        self.predictions_dir = \
            f'{self.run_dir}/predictions/pred_{offset[0]}{offset[1]}_{fold}'
        self.predictions_spinup_dir = \
            f'{self.run_dir}/predictions_spinup/pred_{offset[0]}{offset[1]}_{fold}'
        self.predictions_all_dir = \
            f'{self.run_dir}/predictions_all/pred_{offset[0]}{offset[1]}_{fold}'

        self.has_fold = True

    def set_config(self, config):
        """Set experiment configuration.

        Parameters
        ----------
        config: dict
            Configuration dictionary.
        """
        self.config = config
        self.has_config = True
        return config

    def get_trial_config_and_chkp(self, fold, offset, metric, minimize=True):
        """Get model config and checkpoint for a specific trial.

        Parameters
        ----------
        fold: int
            Fold number.
        offset: tuple of ints
            A two element tuple with the offsets.

        Returns
        -------
        Tuple: checkpoint file (str), configuration (dict)

        """

        exp = Analysis(self.run_dir)
        best_config = exp.get_best_config(
            metric, mode='min' if minimize else 'max')

        search_pattern = os.path.join(
            self.run_dir, f'Trainable/*fold={fold}*offset=({offset[0]}, {offset[1]})*')
        trial_dir = glob2.glob(search_pattern)
        if len(trial_dir) != 1:
            raise AssertionError(
                'None or multiple trial dirs found for the search pattern '
                f'`{search_pattern}`: {trial_dir}'
            )
        trial_dir = trial_dir[0]
        chkpt_file = glob2.glob(os.path.join(
            trial_dir, 'checkpoint_*/chkp.pt'))
        if len(chkpt_file) == 0:
            raise ValueError(f'no checkpoint found for run {trial_dir}.')
        chkpt_file = min(chkpt_file, key=os.path.getctime)

        return chkpt_file, best_config

    def get_best_from_trainable(self, metric, minimize=True):
        """Get best model config and checkpoint file path from Trainables.

        Parameters
        ----------
        metric: str
            Metric that is minimized / maximized.
        minimize: bool
            If `True` (default), the lowest value of `metric` is
            taken, else tha highest.

        Returns
        -------
        Tuple: checkpoint file (str), configuration (dict)
        """

        # Make sure that we do not accidentally have multiple runs in one dir.
        search_pattern = os.path.join(self.run_dir, '*/experiment_state*.json')
        state_file = glob2.glob(search_pattern)
        if len(state_file) != 1:
            raise ValueError(
                f'cannot load experiment state file as the number of matching files is {len(state_file)} != 1.'
                f'\nSearch pattern: {search_pattern}')

        exp = Analysis(self.run_dir, default_metric=metric, default_mode='min')
        best_dir = exp.get_best_logdir(metric, mode='min' if minimize else 'max')
        best_config = exp.get_best_config(metric, mode='min' if minimize else 'max')

        chkpt_file = glob2.glob(os.path.join(best_dir, 'checkpoint_*/chkp.pt'))
        if len(chkpt_file) == 0:
            raise ValueError(f'no checkpoint found for run {best_dir}.')

        chkpt_file = min(chkpt_file, key=os.path.getctime)

        return chkpt_file, best_config

    def get_best_from_summary(self):
        """Load best model configuration from summary dir.

        Returns
        -------
        A dictionary of the configuration.
        """

        best_file = self.best_hp_config_file
        if not os.path.isfile(best_file):
            raise ValueError(
                'tried to load best model config, file does not exist:\n'
                f'{best_file}\nRun `summarize_results.py` to create '
                'such a file.'
            )

        with open(best_file, 'rb') as f:
            config = pickle.load(f)

        return config

    def get_analysis(self, metric):
        """Get ray.tune.Analysis object.

        Returns
        -------
        ray.tune.Analysis object.
        """

        return Analysis(self.run_dir, default_metric=metric, default_mode='min')

    def get_experiment_state_file(self):
        """Get experiment state file.

        Returns
        -------
        Path (str) to a ray.tune experiment state file, .json.

        """
        exp_state_file = glob2.glob(os.path.join(
            self.run_dir, '*', 'experiment_state-*.json'))

        if len(exp_state_file) > 1:
            raise AssertionError(
                'Cannot summarize runs - more than on experiment '
                f'state file found in {self.run_dir}/*')

        return exp_state_file[0]

    def save(self, is_run=False):
        """Save this ExperimentManage along with a configuration file.

        The configuration is not run specific but experiment specific,
        i.e., the search space parameters are not meant to be saved here
        as they change per run.

        Parameters
        ----------
        config: dict
            Configuration dictionary.
        """

        if not self.has_config:
            raise AssertionError(
                'before saving this ExpManager, you must assign a configuration '
                'using `myExpManager.set_config()`.'
            )

        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.expconfig_file, 'wb') as f:
            pickle.dump(self, f)

    def save_searchspace(self, space):
        """Save string representation of search space to file.

        Parameters
        ----------
        space: ConfigSpace
            Search space from package ConfigSpace.

        """
        with open(self.searchspace_file, mode='w+') as f:
            f.write(str(space))

    def save_config(self, config=None):
        """Save json representation of configuration to file.

        Parameters
        ----------
        config: dict
            Dict of experiment configuration. If no config is passed, the
            configuration attribute `self.config` is used, which then must
            have been set beforehand using `myExpManager.set_config`.

        """
        if config is None:
            config = self.config
        with open(self.config_file, mode='w+') as f:
            json.dump(config, f, sort_keys=True, indent=4)

    @property
    def config(self):
        if not self.has_config:
            raise AttributeError(
                'the config attribute is not yet set. Use '
                '`myExpManager.set_config(...)`.'
            )
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def store(self):
        return self._store

    @store.setter
    def store(self, value):
        if not isinstance(value, str):
            raise TypeError(
                '`store` must be of type `str` but is type '
                f'`{type(value).__name__}`.'
            )
        self._store = value

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, value):
        if not isinstance(value, str):
            raise TypeError(
                '`experiment` must be of type `str` but is type '
                f'`{type(value).__name__}`.'
            )
        self._experiment = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError(
                '`name` must be of type `str` but is type '
                f'`{type(value).__name__}`.'
            )
        self._name = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ('tune', 'cv'):
            raise ValueError(
                f'`mode` must be one of (`tune`, `cv`) but is `{value}`.'
            )
        self._mode = value

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, value):
        if not isinstance(value, int):
            raise TypeError(
                '`fold` must be an integer but is type '
                f' `{type(value).__name__}`.'
            )
        self._fold = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if not hasattr(value, '__iter__') or len(value) != 2 or \
           not isinstance(value[0], int) or \
           not isinstance(value[1], int):
            raise ValueError(
                    '`offset` must be an iterable of two integers but '
                    f'is `{value}`.'
            )
        self._offset = value

    def __str__(self):
        s = (
            '<ExpManager>\n'
            f'  path\n    {self.run_dir}\n'
            f'  summary\n    {self.summary_dir}\n'
        )
        if self.has_fold:
            s += (
                f'  predictions\n    {self.predictions_dir}\n'
            )
        return s

    def __repr__(self):
        return self.__str__()
