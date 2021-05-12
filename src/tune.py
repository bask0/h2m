import ray
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.logger import JsonLogger, CSVLogger, TBXLogger
import numpy as np
import os
from typing import Dict, Any
import logging
import argparse
import shutil
import pickle
import torch

# Own classes and methods.
from dataset import Bucket, getDataloader, check_bucket
from models.hybridmodel_loop import HybridModel
from utils.trainer_jit import Trainer
from utils.summarize_runs import summarize
from utils.merge_predictions import merge_predictions
from experiment_config import get_config
from utils.experimentmanager import ExpManager
from utils.helpers import get_max_concurrent, set_seed, trial_str_creator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TUNE_DISABLE_AUTO_INIT'] = '1'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    DEVICE = 'cpu'

# CV_FOLDS = [0, 1, 2, 3, 4]
# CV_OFFSETS = [(0, 1), (1, 0), (1, 1)]

# warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args() -> Dict[str, Any]:
    """Parse arguments.

    Returns
    --------
    Dict of arguments.

    """
    parser = argparse.ArgumentParser()

    def parse_bool(v):
        """Parse boolean if no action is used (to force decision)."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('true'):
            return True
        elif v.lower() in ('false'):
            return False
        else:
            raise argparse.ArgumentTypeError(
                'Boolean value expected (True | False).')

    parser.add_argument(
        '-e',
        '--experiment',
        type=str,
        help='Experiment name, one of (multitask (default)).',
        default='hybrid'
    )

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        help=(
            'Run name, one of (all_vars_no_task_weighting (default) | '
            'all_vars_task_weighting | {task_name}_no_task_weighting).'),
        default='all_vars_task_weighting'
    )

    parser.add_argument(
        '--tune',
        type=str,
        help='Wheter to tune hyperparameters (true) or tune model (false).',
        required=True
    )

    parser.add_argument(
        '--use_default_config',
        action='store_true',
        help=(
            'If true, run model on default configuration '
            'instead of using best model run.')
    )

    parser.add_argument(
        '--predict',
        action='store_true',
        help='Predict from best run.'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training.'
    )

    parser.add_argument(
        '--small_aoi',
        action='store_true',
        help='Reduce AOI to .'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run experiment with very short training / validation cycles for debugging.'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debugging mode, print confusing stuff.'
    )

    args, _ = parser.parse_known_args()
    args.tune = parse_bool(args.tune)

    os.environ['debug'] = 'True' if args.debug else 'False'

    global CV_FOLDS
    global CV_OFFSETS
    if args.test:
        CV_FOLDS = [0]
        CV_OFFSETS = [(0, 1)]
    else:
        CV_FOLDS = [0, 1, 2, 3, 4]
        CV_OFFSETS = [(0, 1), (1, 0), (1, 1)]

    return args


class Trainable(ray.tune.Trainable):
    """"Model trainer called by ray.tunes."""

    def _setup(self, config):

        logging.basicConfig(
            format='%(asctime)s-%(levelname)s: %(message)s',
            datefmt='%d-%b-%y %H:%M:%S',
            level=logging.INFO)

        self.emanager = ExpManager.load_from_path(
            config.pop('env')['emanager_restore_path'])

        self.emanager.set_fold_paths(config['fold'], config['offset'])

        bucket = Bucket(
            self.emanager.config['dataset'],
            overwrite=False,
            read_only=True,
            sample_formatter_path=self.emanager.config['dataconfig'],
            use_msc_spinup=False)

        bucket.set_cv_fold_split(fold=config.get('fold'), latoffset=config.get(
            'offset')[0], lonoffset=config.get('offset')[1], n_sets=6)

        bucket.set_rep_years(
            self.emanager.config['num_spinup_years'], ref_var='tair')

        check_bucket(bucket, self.emanager.config['tasks'])

        self.train_loader = getDataloader(
            dataset=bucket,
            cv_set='train',
            batch_size=self.emanager.config['batch_size'],
            seed=self.emanager.config['seed'],
            nworkers=self.emanager.config['num_workers'],
            n_rep_years=self.emanager.config['num_spinup_years'],
            pin_memory=self.emanager.config['pin_memory'])

        self.valid_loader = getDataloader(
            dataset=bucket,
            cv_set='valid',
            batch_size=self.emanager.config['batch_size'],
            seed=self.emanager.config['seed'],
            nworkers=self.emanager.config['num_workers'],
            n_rep_years=self.emanager.config['num_spinup_years'],
            pin_memory=self.emanager.config['pin_memory'])

        self.test_loader = getDataloader(
            dataset=bucket,
            cv_set='test',
            batch_size=self.emanager.config['batch_size'],
            seed=self.emanager.config['seed'],
            nworkers=self.emanager.config['num_workers'],
            n_rep_years=self.emanager.config['num_spinup_years'],
            pin_memory=self.emanager.config['pin_memory'])

        self.all_loader = getDataloader(
            dataset=bucket,
            cv_set='all',
            batch_size=self.emanager.config['batch_size'],
            seed=self.emanager.config['seed'],
            nworkers=self.emanager.config['num_workers'],
            n_rep_years=self.emanager.config['num_spinup_years'],
            pin_memory=self.emanager.config['pin_memory'])

        self.model = HybridModel(
            num_features=3,
            static_hidden_size=int(config.get('stat_nh')),
            static_num_layers=int(config.get('stat_nl')),
            static_enc_size=self.emanager.config['num_stat_enc'],
            static_dropout=np.round(config.get('stat_dropout'), 2),
            lstm_hidden_size=int(config.get('temp_nh')),
            task_hidden_size=int(config.get('task_nh')),
            task_num_layers=int(config.get('task_nl')),
            task_dropout=np.round(config.get('task_dropout'), 2))

        self.model.to_device(DEVICE)

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            test_loader=self.test_loader,
            all_loader=self.all_loader,
            learning_rate=config.get('lr', 0.01),
            learning_rate_taskw=config.get('lr_taskw', None),
            weight_decay=config.get('w_decay', 0.01),
            task_weighting=self.emanager.config['task_weighting'],
            tasks=self.emanager.config['tasks'],
            gradient_clipping=config.get('grad_clip', None),
            device=DEVICE,
            logdir=self._logdir
        )

    def _train(self):

        stats = self.trainer.train_epoch(
            self.emanager.config['num_train_batches'])
        stats_eval = self.trainer.eval_epoch()
        stats.update(stats_eval)

        # Only do eaerly stop after grace period.
        if stats['epoch'] < self.emanager.config['grace_period']:
            stats['patience_counter'] = -1

        return stats

    def _stop(self):

        if not self.emanager.tune:
            self.predict()
            merge_predictions(
                path=self.emanager.expconfig_file,
                fold=self.emanager.fold,
                offset=self.emanager.offset
            )
            merge_predictions(
                path=self.emanager.expconfig_file,
                fold=self.emanager.fold,
                offset=self.emanager.offset,
                is_spinup=True
            )
            self.predict(predict_all=True)
            merge_predictions(
                path=self.emanager.expconfig_file,
                fold=self.emanager.fold,
                offset=self.emanager.offset,
                is_all=True
            )

    def predict(self, restore_from_best=True, predict_all=False):

        print('Predicting...')

        dump_dir = self.emanager.predictions_all_dir if predict_all else self.emanager.predictions_dir
        dump_dir_spinup = self.emanager.predictions_spinup_dir

        if os.path.isdir(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir, exist_ok=False)

        if not predict_all:
            if os.path.isdir(dump_dir_spinup):
                shutil.rmtree(dump_dir_spinup)
            os.makedirs(dump_dir_spinup, exist_ok=False)

        # Save training, validation and test mask.
        masks = self.trainer.train_loader.dataset.get_masks()
        masks.to_netcdf(os.path.join(
            dump_dir, f'cv_masks.nc'))

        if restore_from_best:
            checkp, _ = self.emanager.get_trial_config_and_chkp(
                self.emanager.fold, self.emanager.offset, self.emanager.config['metric'])
            self._restore(checkp)

        stats_test, pred, spinup_pred = self.trainer.predict(predict_all)

        dump_file = os.path.join(
            dump_dir,
            f'pred_{np.random.choice(999999999999):010d}.pck')
        print('  saving to: ', dump_file)

        if predict_all:
            sampler = self.all_loader
        else:
            sampler = self.valid_loader

        with open(dump_file, mode='wb') as f:
            pickle.dump(
                dict(
                    pred=pred,
                    daily_bin=sampler.dataset.get_bins('swe')
                ), f
            )

        if not predict_all:
            dump_file_spinup = os.path.join(
                dump_dir_spinup,
                f'pred_{np.random.choice(999999999999):010d}.pck')

            with open(dump_file_spinup, mode='wb') as f:
                pickle.dump(
                    dict(
                        pred=spinup_pred,
                        daily_bin=[
                            0, self.emanager.config['num_spinup_years'] * 365]
                    ), f
                )

    def _save(self, path):
        path = self.trainer.save(path)

        return path

    def _restore(self, path):
        print(f'restoring model from: {path}')

        self.trainer.restore(path)


def tune_parameters(
        tune, experiment='hybrid', name='all_vars_task_weighting', use_default_config=False,
        predict=False, resume=False, small_aoi=False, test=False, debug=False):

    if resume and predict:
        raise ValueError('cannot set flags `--resume` and `--predict` at the same time.')

    config, space = get_config(experiment, name)

    set_seed(config['seed'])
    max_concurrent = get_max_concurrent(config['ngpu'])

    bobh_search = TuneBOHB(
        space=space,
        max_concurrent=max_concurrent,
        metric=config['metric'],
        mode=config['mode']
    )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='epoch',
        metric=config['metric'],
        mode=config['mode'],
        max_t=config['max_t'],
        reduction_factor=config['reduction_factor'])

    overwrite = not (predict or resume)
    emanager = ExpManager(
        config['store'], experiment, name,
        'tune' if tune else 'cv', overwrite=overwrite)
    config = emanager.set_config(config)

    # Save human-readable search space and configuration for this experiment.
    emanager.save_searchspace(space)
    emanager.save_config()

    # Save machine-readable ExpManager object. This is used by the Trainable to retieve
    # the configuration (prefered here as it provides a clean way to separate search space
    # and experiment configuration).
    emanager.save()

    # Predict based on best run parameters, iterate folds & offsets.
    if not tune and not predict:
        if use_default_config:
            best_config = {}
        else:
            best_config = emanager.get_best_from_summary()

        # With fixed HPs, we iterate folds and offsets.
        best_config.update({"fold": ray.tune.grid_search(CV_FOLDS)})
        best_config.update({"offset": ray.tune.grid_search(CV_OFFSETS)})
        best_config.update({'env': {'emanager_restore_path': emanager.expconfig_file}})

        ray.tune.run(
            Trainable,
            resources_per_trial={
                'cpu': emanager.config['ncpu'],
                'gpu': emanager.config['ngpu'],
            },
            num_samples=1,
            config=best_config,
            local_dir=emanager.run_dir,
            raise_on_failed_trial=False,
            trial_name_creator=trial_str_creator,
            verbose=1,
            with_server=False,
            #ray_auto_init=False,
            loggers=[JsonLogger, CSVLogger, TBXLogger],
            checkpoint_at_end=True,
            checkpoint_freq=1,
            checkpoint_score_attr='min-' + emanager.config['metric'],
            keep_checkpoints_num=1,
            reuse_actors=False,
            resume=resume,
            stop={
                'patience_counter': emanager.config['patience'],
                'epoch': 40 if test else 999999999
            }
        )

        # Avoid error (ray.tune bug?).
        import time
        time.sleep(1000)

    # Run hyperopt.
    elif not predict:
        ray.tune.run(
            Trainable,
            resources_per_trial={
                'cpu': config['ncpu'],
                'gpu': config['ngpu'],
            },
            config={'env': {'emanager_restore_path': emanager.expconfig_file}},
            num_samples=2 if test else config['num_samples'],
            local_dir=emanager.run_dir,
            raise_on_failed_trial=False,
            trial_name_creator=trial_str_creator,
            verbose=1,
            with_server=False,
            # ray_auto_init=True,
            search_alg=bobh_search,
            scheduler=bohb_scheduler,
            loggers=[JsonLogger, CSVLogger, TBXLogger],
            checkpoint_at_end=True,
            checkpoint_freq=1,
            checkpoint_score_attr='min-' + config['metric'],
            keep_checkpoints_num=1,
            reuse_actors=False,
            resume=resume,
            stop={'patience_counter': config['patience'],
                  'epoch': 5 if test else 999999999}
        )

    # Do predictions.
    if tune:

        checkp, best_config = emanager.get_best_from_trainable(
            metric=config['metric'])
        emanager.set_fold_paths(best_config['fold'], best_config['offset'])

        trainable = Trainable(best_config)
        trainable._restore(checkp)
        trainable.predict(restore_from_best=False)

        merge_predictions(
            path=emanager.expconfig_file,
            fold=emanager.fold,
            offset=emanager.offset)
        merge_predictions(
            path=emanager.expconfig_file,
            fold=emanager.fold,
            offset=emanager.offset,
            is_spinup=True)

    elif predict:

        for offset in CV_OFFSETS:

            for fold in CV_FOLDS:

                emanager.set_fold_paths(
                    fold, offset)

                checkp, best_config = emanager.get_trial_config_and_chkp(
                    fold, offset, config['metric'])

                best_config['offset'] = offset
                best_config['fold'] = fold

                trainable = Trainable(best_config)
                trainable._restore(checkp)
                trainable.predict(restore_from_best=False)

                merge_predictions(
                    path=emanager.expconfig_file,
                    fold=emanager.fold,
                    offset=emanager.offset)
                merge_predictions(
                    path=emanager.expconfig_file,
                    fold=emanager.fold,
                    offset=emanager.offset,
                    is_spinup=True)

    summarize(
        path=emanager.expconfig_file,
        overwrite=True)

    # Postprocess predictions.
    #if not tune:
    #    postprocess(offsets=CV_OFFSETS, folds=CV_FOLDS)
    #    postprocess(offsets=CV_OFFSETS, folds=CV_FOLDS, is_all=True)


if __name__ == '__main__':

    try:
        args = parse_args()

        ray.init(include_dashboard=True, object_store_memory=int(
            15e9), num_cpus=80, dashboard_port=8442, dashboard_host='0.0.0.0')
        print(ray.nodes())
        tune_parameters(args.tune, experiment=args.experiment, name=args.name,
                        use_default_config=args.use_default_config, predict=args.predict,
                        resume=args.resume, small_aoi=args.small_aoi,
                        test=args.test, debug=args.debug)
        ray.shutdown()

    except KeyboardInterrupt:
        print('\n*** Keyboard interrupt, shutting down... ***')
        ray.shutdown()
