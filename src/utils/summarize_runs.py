"""
Analyze a ray.tune run; plot training curves, show best run etc.
"""

import matplotlib.gridspec as gridspec
import argparse
from typing import Dict, Any
import os
import shutil
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt

from utils.experimentmanager import ExpManager

TASKS = ['et', 'swe', 'q', 'tws']
CONSTRS = ['cwd']


def parse_args() -> Dict[str, Any]:
    """Parse arguments.

    Returns
    --------
    Dict of arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            'Summarize ray.tune runs from a given experiment state file (.json). A directory called '
            '`summary` is created in the directory containing the .json file.'
        ))

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='path to experiment manager file'
    )

    parser.add_argument(
        '-t',
        '--train_metric',
        type=str,
        default='uloss_train',
        help='the train metric name'
    )

    parser.add_argument(
        '-e',
        '--eval_metric',
        type=str,
        default='uloss_valid',
        help='the evaluation metric name (also metric that is used to find best run)'
    )

    parser.add_argument(
        '-O',
        '--overwrite',
        action='store_true',
        help='whether to overwrite existing out_dir, default is `false`'
    )

    args = parser.parse_args()

    return args


def loss_name(task, is_constr=False):
    if is_constr:
        constr_str = '_constr'
    else:
        constr_str = ''

    uloss_train = task + constr_str + '_uloss' + '_train'
    uloss_valid = task + constr_str + '_uloss' + '_valid'
    wloss_train = task + constr_str + '_wloss' + '_train'
    wloss_valid = task + constr_str + '_wloss' + '_valid'
    weight_train = task + constr_str + '_weight' + '_train'
    weight_valid = task + constr_str + '_weight' + '_valid'

    return uloss_train, uloss_valid, wloss_train, wloss_valid, weight_train, weight_valid


def summarize(
        path: str,
        train_metric: str = 'uloss_train',
        eval_metric: str = 'uloss_valid',
        overwrite: bool = False,
        return_analysis: bool = False) -> None:

    em = ExpManager.load_from_path(path)

    print(f'\nLoading experiment state file loaded from:  \n{em.run_dir}\n')

    summary_dir = em.summary_save_dir

    if os.path.isdir(summary_dir):
        if not overwrite:
            raise ValueError(f'Target directory `{summary_dir}` exists, use `--overwrite` to replace.')
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)

    exp = em.get_analysis(metric=eval_metric)

    configs = exp.dataframe()
    configs['rundir'] = [os.path.join(l, 'progress.csv')
                         for l in configs['logdir']]
    runs = []
    for i, f in enumerate(configs['rundir']):
        df = pd.read_csv(f)
        df['uid'] = i
        runs.append(df)
    runs = pd.concat(runs)

    best_run_dir = exp.get_best_logdir(eval_metric, mode='min')
    best_run_file = os.path.join(best_run_dir, 'progress.csv')
    best_run = df = pd.read_csv(best_run_file)

    print(f'Best run ID: {best_run_dir}')

    for f in ['json', 'pkl']:
        in_file = os.path.join(best_run_dir, f'params.{f}')
        out_file = os.path.join(summary_dir, f'best_params.{f}')
        copyfile(in_file, out_file)

    # Plot runs.
    plot_all(runs, os.path.join(summary_dir, 'all_runs.png'))
    plot_single(best_run, os.path.join(summary_dir, 'best_run.png'))


def plot_all(
        runs: pd.core.frame.DataFrame,
        savepath: str,
        log_scale=False) -> None:

    nrows = len(TASKS) + len(CONSTRS) + 1
    f = plt.figure(figsize=(15, nrows * 2))

    gs0 = gridspec.GridSpec(1, 2, figure=f, wspace=0.3)

    gs00 = gs0[0].subgridspec(nrows, 2, wspace=0.05, hspace=0.05)
    axes0 = [f.add_subplot(gs00[i, 0]) for i in range(nrows)]
    axes1 = [f.add_subplot(gs00[i, 1]) for i in range(nrows)]

    gs01 = gs0[1].subgridspec(nrows, 2, wspace=0.05, hspace=0.05)
    axes2 = [f.add_subplot(gs01[i, 0]) for i in range(nrows)]
    axes3 = [f.add_subplot(gs01[i, 1]) for i in range(nrows)]

    axes0[0].set_title('train')
    axes1[0].set_title('valid')
    axes2[0].set_title('train')
    axes3[0].set_title('valid')

    axes0[0].text(
        1.0, 1.1, 'unweighted', transform=axes0[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=14)
    axes2[0].text(
        1.0, 1.1, 'weighted', transform=axes2[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=14)

    runs.groupby(['uid']).plot(
        x='epoch', y='uloss_train',
        ax=axes0[0], legend=False, logy=log_scale)
    axes0[0].set_ylabel('total loss', size=15)
    runs.groupby(['uid']).plot(
        x='epoch', y='uloss_valid',
        ax=axes1[0], legend=False, logy=log_scale)

    runs.groupby(['uid']).plot(
        x='epoch', y='wloss_train',
        ax=axes2[0], legend=False, logy=log_scale)
    axes2[0].set_ylabel('total loss', size=15)
    runs.groupby(['uid']).plot(
        x='epoch', y='wloss_valid',
        ax=axes3[0], legend=False, logy=log_scale)

    for i, t in enumerate(TASKS):
        uloss_train, uloss_valid, wloss_train, wloss_valid, weight_train, weight_valid = loss_name(
            t)
        runs.groupby(['uid']).plot(x='epoch', y=uloss_train,
                                   ax=axes0[i+1], legend=False, logy=log_scale)
        axes0[i+1].set_ylabel(t, size=15)
        runs.groupby(['uid']).plot(x='epoch', y=uloss_valid,
                                   ax=axes1[i+1], legend=False, logy=log_scale)

        runs.groupby(['uid']).plot(x='epoch', y=wloss_train,
                                   ax=axes2[i+1], legend=False, logy=log_scale)
        axes2[i+1].set_ylabel(t, size=15)
        runs.groupby(['uid']).plot(x='epoch', y=wloss_valid,
                                   ax=axes3[i+1], legend=False, logy=log_scale)

    for i, t in enumerate(CONSTRS):
        uloss_train, uloss_valid, wloss_train, wloss_valid, weight_train, weight_valid = loss_name(
            t, True)
        runs.groupby(['uid']).plot(x='epoch', y=uloss_train,
                                   ax=axes0[i+5], legend=False, logy=log_scale)
        axes0[i+5].set_ylabel(t, size=15)
        runs.groupby(['uid']).plot(x='epoch', y=uloss_valid,
                                   ax=axes1[i+5], legend=False, logy=log_scale)

        runs.groupby(['uid']).plot(x='epoch', y=wloss_train,
                                   ax=axes2[i+5], legend=False, logy=log_scale)
        axes2[i+5].set_ylabel(t, size=15)
        runs.groupby(['uid']).plot(x='epoch', y=wloss_valid,
                                   ax=axes3[i+5], legend=False, logy=log_scale)

    for i in range(nrows):
        ylim_0 = axes0[i].get_ylim()
        ylim_1 = axes1[i].get_ylim()
        ylim_2 = axes2[i].get_ylim()
        ylim_3 = axes3[i].get_ylim()

        ylim_01 = (min(ylim_0[0], ylim_1[0]), max(ylim_0[1], ylim_1[1])*0.5)
        ylim_23 = (min(ylim_2[0], ylim_3[0]), max(ylim_2[1], ylim_3[1])*0.5)

        axes0[i].set_ylim(ylim_01)
        axes1[i].set_ylim(ylim_01)
        axes2[i].set_ylim(ylim_23)
        axes3[i].set_ylim(ylim_23)

        axes0[i].set_xlabel('')
        axes1[i].set_xlabel('')
        axes2[i].set_xlabel('')
        axes3[i].set_xlabel('')

        axes1[i].set_ylabel('')
        axes3[i].set_ylabel('')

        axes1[i].tick_params(labelleft=False, which='both')
        axes3[i].tick_params(labelleft=False, which='both')

        axes0[i].patch.set_facecolor('white')
        axes1[i].patch.set_facecolor('white')
        axes2[i].patch.set_facecolor('white')
        axes3[i].patch.set_facecolor('white')

        if i < (nrows-1):
            axes0[i].set_xticks([])
            axes1[i].set_xticks([])
            axes2[i].set_xticks([])
            axes3[i].set_xticks([])

    f.align_ylabels(axes0)
    f.align_ylabels(axes2)

    f.patch.set_alpha(0)
    f.savefig(savepath, bbox_inches='tight', dpi=300, transparent=True)


def plot_single(
        single_run: pd.core.frame.DataFrame,
        savepath: str,
        log_scale=False) -> None:

    nrows = len(TASKS) + len(CONSTRS) + 1
    f = plt.figure(figsize=(15, nrows * 2))

    gs0 = gridspec.GridSpec(1, 3, figure=f, wspace=0.3)

    gs00 = gs0[0].subgridspec(nrows, 2, wspace=0.05, hspace=0.05)
    axes0 = [f.add_subplot(gs00[i, 0]) for i in range(nrows)]
    axes1 = [f.add_subplot(gs00[i, 1]) for i in range(nrows)]

    gs01 = gs0[1].subgridspec(nrows, 2, wspace=0.05, hspace=0.05)
    axes2 = [f.add_subplot(gs01[i, 0]) for i in range(nrows)]
    axes3 = [f.add_subplot(gs01[i, 1]) for i in range(nrows)]

    gs02 = f.add_subplot(gs0[2])

    axes0[0].set_title('train')
    axes1[0].set_title('valid')
    axes2[0].set_title('train')
    axes3[0].set_title('valid')

    axes0[0].text(
        1.0, 1.1, 'unweighted', transform=axes0[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=14)
    axes2[0].text(
        1.0, 1.1, 'weighted', transform=axes2[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=14)
    gs02.text(
        0.5, 1.01, 'weights', transform=gs02.transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold', fontsize=14)

    single_run.plot(x='epoch', y='uloss_train',
                    ax=axes0[0], legend=False, logy=log_scale)
    axes0[0].set_ylabel('total loss', size=15)
    single_run.plot(x='epoch', y='uloss_valid',
                    ax=axes1[0], legend=False, logy=log_scale)

    single_run.plot(x='epoch', y='wloss_train',
                    ax=axes2[0], legend=False, logy=log_scale)
    axes2[0].set_ylabel('total loss', size=15)
    single_run.plot(x='epoch', y='wloss_valid',
                    ax=axes3[0], legend=False, logy=log_scale)

    for i, t in enumerate(TASKS):
        uloss_train, uloss_valid, wloss_train, wloss_valid, weight_train, weight_valid = loss_name(
            t)
        single_run.plot(x='epoch', y=uloss_train,
                        ax=axes0[i+1], legend=False, logy=log_scale)
        axes0[i+1].set_ylabel(t, size=15)
        single_run.plot(x='epoch', y=uloss_valid,
                        ax=axes1[i+1], legend=False, logy=log_scale)

        single_run.plot(x='epoch', y=wloss_train,
                        ax=axes2[i+1], legend=False, logy=log_scale)
        axes2[i+1].set_ylabel(t, size=15)
        single_run.plot(x='epoch', y=wloss_valid,
                        ax=axes3[i+1], legend=False, logy=log_scale)

        single_run.plot(x='epoch', y=weight_train,
                        ax=gs02, legend=True, label=t)

    for i, t in enumerate(CONSTRS):
        uloss_train, uloss_valid, wloss_train, wloss_valid, weight_train, weight_valid = loss_name(
            t, True)
        single_run.plot(x='epoch', y=uloss_train,
                        ax=axes0[i+5], legend=False, logy=log_scale)
        axes0[i+5].set_ylabel(t, size=15)
        single_run.plot(x='epoch', y=uloss_valid,
                        ax=axes1[i+5], legend=False, logy=log_scale)

        single_run.plot(x='epoch', y=wloss_train,
                        ax=axes2[i+5], legend=False, logy=log_scale)
        axes2[i+5].set_ylabel(t, size=15)
        single_run.plot(x='epoch', y=wloss_valid,
                        ax=axes3[i+5], legend=False, logy=log_scale)
        single_run.plot(x='epoch', y=weight_train,
                        ax=gs02, legend=True, label=t)

    for i in range(nrows):
        ylim_0 = axes0[i].get_ylim()
        ylim_1 = axes1[i].get_ylim()
        ylim_2 = axes2[i].get_ylim()
        ylim_3 = axes3[i].get_ylim()

        ylim_01 = (min(ylim_0[0], ylim_1[0]), max(ylim_0[1], ylim_1[1]))
        ylim_23 = (min(ylim_2[0], ylim_3[0]), max(ylim_2[1], ylim_3[1]))

        axes0[i].set_ylim(ylim_01)
        axes1[i].set_ylim(ylim_01)
        axes2[i].set_ylim(ylim_23)
        axes3[i].set_ylim(ylim_23)

        axes0[i].set_xlabel('')
        axes1[i].set_xlabel('')
        axes2[i].set_xlabel('')
        axes3[i].set_xlabel('')

        axes1[i].set_ylabel('')
        axes3[i].set_ylabel('')

        axes1[i].tick_params(labelleft=False, which='both')
        axes3[i].tick_params(labelleft=False, which='both')

        axes0[i].patch.set_facecolor('white')
        axes1[i].patch.set_facecolor('white')
        axes2[i].patch.set_facecolor('white')
        axes3[i].patch.set_facecolor('white')

        if i < (nrows-1):
            axes0[i].set_xticks([])
            axes1[i].set_xticks([])
            axes2[i].set_xticks([])
            axes3[i].set_xticks([])

    gs02.set_ylabel('task weight w', size=15)
    gs02.patch.set_facecolor('white')

    f.align_ylabels(axes0)
    f.align_ylabels(axes2)

    f.patch.set_alpha(0)
    f.savefig(savepath, bbox_inches='tight', dpi=300, transparent=True)


if __name__ == '__main__':

    args = parse_args()

    summarize(
        args.path,
        args.train_metric,
        args.eval_metric,
        args.overwrite)
