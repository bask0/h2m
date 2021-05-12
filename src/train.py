
import warnings
import os

import time

from tune import getDataloader

from dataset import Bucket

import torch
from models.hybridmodel_loop_jit import HybridModel
from utils.trainer_jit import Trainer
from experiment_config import get_config

os.environ['debug'] = 'False'

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print('DEVICE: ', DEVICE)

warnings.simplefilter(action='ignore', category=FutureWarning)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{method.__name__} {(te - ts)} s')
        return result
    return timed


fix_config, _ = get_config('hybrid', 'all_vars_task_weighting')

config = {}

bucket = Bucket(
    fix_config['dataset'],
    overwrite=False,
    read_only=True,
    sample_formatter_path=fix_config['dataconfig'])

bucket.set_cv_fold_split(fold=0, latoffset=0, lonoffset=0, n_sets=5)

bucket.set_rep_years(
    int(config.get('num_spinup_years', 10)), ref_var='tair')

pin_memory = False

train_loader = getDataloader(
    dataset=bucket,
    cv_set='train',
    batch_size=100,
    seed=fix_config['seed'],
    nworkers=5,
    n_rep_years=config.get('num_spinup_years', 10),
    pin_memory=pin_memory)

valid_loader = getDataloader(
    dataset=bucket,
    cv_set='valid',
    batch_size=100,
    seed=fix_config['seed'],
    nworkers=5,
    n_rep_years=config.get('num_spinup_years', 10),
    pin_memory=pin_memory)

valid_loader = getDataloader(
    dataset=bucket,
    cv_set='valid',
    batch_size=100,
    seed=fix_config['seed'],
    nworkers=5,
    n_rep_years=config.get('num_spinup_years', 10),
    pin_memory=pin_memory)

model = HybridModel(
    num_features=3,
    static_hidden_size=64,
    static_num_layers=2,
    static_enc_size=12,
    static_dropout=0.1,
    lstm_hidden_size=128,
    task_hidden_size=64,
    task_num_layers=2,
    task_dropout=0.1)

model.to_device(DEVICE)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=valid_loader,
    learning_rate=0.001,
    learning_rate_taskw=0.001,
    weight_decay=0.001,
    task_weighting=True,
    tasks=fix_config['tasks'],
    gradient_clipping=None,
    device=DEVICE,
    logdir='/tmp/ray_test/'
)


@timeit
def run_training(n=1):
    print('Training epoch')
    for i in range(n):
        trainer.train_epoch()


@timeit
def run_validation(n=1):
    print('Validation epoch')
    for i in range(n):
        trainer.eval_epoch()


run_training(1)
run_validation(1)

print('Done')
