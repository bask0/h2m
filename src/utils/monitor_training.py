import argparse
from ray.tune.analysis import Analysis
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from time import sleep
import os
from datetime import datetime
from matplotlib import cm

cmap = cm.get_cmap('tab20b')
NCOLORS = len(cmap.colors)
COLORS = [
    f'rgb({cmap(i)[0]}, {cmap(i)[1]}, {cmap(i)[2]})' for i in np.arange(NCOLORS)]


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Monitor model training.')
    parser.add_argument('-p', '--path', type=str,
                        help='path to an ray results directory.')
    parser.add_argument('-o', '--save_path', type=str,
                        help='temporary html file to store plot', default='/workspace/hydrodl/src/tmp/monitor.html')
    parser.add_argument('-f', '--update_freq', type=int, default=3,
                        help='update frequency in miutes, default is 3')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='verbosity, 0=silent, 1=verboe.')
    parser.add_argument('-m', '--min_epochs', type=int, default=2,
                        help='runs with num epochs below this threshold are not displayed.')

    return parser


def print_status(s):
    print(
        f'\033[92m{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\033[0m --- {s}')


def make_plot(file, save_path, min_epochs, verbosity):

    if save_path[-5:] != '.html':
        raise ValueError(
            'argument `save_path` must be a `.html` file.'
        )
    if verbosity not in [0, 1]:
        raise ValueError(
            'argument `save_path` must be an integer `0` or `1`'
        )

    es = Analysis(file)

    df = []
    for i, (k, v) in enumerate(es.trial_dataframes.items()):
        v['uid'] = i
        df.append(v)

    if len(df) == 0:
        print_status(f'\033[93mwaiting for data...\033[0m')
        return 0

    df = pd.concat(df)

    if df['epoch'].max() < min_epochs:
        print_status(
            f'\033[93mwaiting for data (no run with `min_epochs>={min_epochs}`)\033[0m')
        return 0

    df_params = es.dataframe()
    cols = [col for col in df_params.columns if 'config/' in col and 'offset' not in col and 'env' not in col]
    df_params = df_params[cols + ['trial_id']]

    df = pd.merge(df, df_params, on='trial_id')

    variables = ['', 'tws_', 'et_', 'swe_', 'q_']
    variables_nice = ['total loss', 'tws', 'et', 'swe', 'q']

    fig = make_subplots(rows=len(variables), cols=3, shared_xaxes='columns', shared_yaxes='rows',
                        subplot_titles=['training', 'validation', 'weights'], row_titles=variables_nice,
                        horizontal_spacing=0.02, vertical_spacing=0.01)

    for var_i, var in enumerate(variables):
        for uid_i, uid in enumerate(np.unique(df['uid'])):
            color = COLORS[uid_i % NCOLORS]
            df_ = df.loc[df['uid'] == uid, :]
            hoverdata = df_[cols].iloc[0, :]
            hovertext = ''

            for k, v in hoverdata.items():
                if k != 'env':
                    hovertext += f"{k.split('/')[1]:20}  {v}<br>"

            if df_['epoch'].max() > 1:
                for cvset_i, cvset in enumerate(['train', 'valid']):

                    y_var = f'{var}uloss_{cvset}'
                    fig.add_trace(go.Scatter(
                        x=df_['epoch'], y=df_[y_var],
                        hovertext=hovertext, mode='lines', line=dict(color=color, width=1)),
                        row=var_i + 1, col=cvset_i + 1
                    )

            if var_i > 0:
                fig.add_trace(go.Scatter(x=df_['epoch'], y=df_[f'{var}weight_valid'], hovertext=hovertext, mode='lines', line=dict(
                    color=color, width=1)), row=var_i + 1, col=3)

    for i in range((len(variables) * 3)):
        I = i + 1
        # xaxis: shared by first and second column, and among third column.
        if (I % 3 == 1) or (I % 3 == 2):
            fig.layout[f'xaxis{I if I > 0 else ""}'].update({'matches': f'x'})
        else:
            fig.layout[f'yaxis{I if I > 0 else ""}'].pop('matches')
            fig.layout[f'yaxis{I if I > 0 else ""}'].update({'showticklabels': True})
            # fig.layout[f'yaxis{I if I > 0 else ""}'].update({'side': 'right'})

    fig.update_layout(title='model training --- updated ' + datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"), showlegend=False, width=1200, height=len(variables)*200)

    fig.write_html(save_path)

    if verbosity:
        print_status(
            f'file updated: \u001b[34m{os.path.abspath(save_path)}\033[0m')


def main(args=None):
    """
    Main entry point for your project.
    Args:
        args : list
            A of arguments as if they were input in the command line. Leave it
            None to use sys.argv.
    """

    parser = get_parser()
    args = parser.parse_args(args)

    try:
        print('\n')
        print_status(
            f'\033[96mmonitoring model run (suppressing runs with num_epochs<{args.min_epochs})\033[0m')

        with open(args.save_path, 'w') as f:
            txt = f'''
                <!DOCTYPE html>
                <html>
                <body>
                <meta http-equiv="refresh" content="60" >
                <h1>Monitoring model run (refreshed: <span id="datetime"></span>)</h1>
                <p>No data found in `{os.path.abspath(args.save_path)}`.
                <button onClick="window.location.reload();">Refresh page</button> to check for new data.
                The page will refresh every {args.update_freq} minute(s) automatically.</p>

                <script>
                var dt = new Date();
                document.getElementById("datetime").innerHTML = dt.toLocaleString();
                </script>

                </body>
                </html>
            '''
            f.write(
                txt
            )

        print_status(
            f'\033[96m{"-"*60}\033[0m')
        while True:
            # Path might change when search pattern is used.
            make_plot(args.path, args.save_path, args.min_epochs, args.verbosity)
            sleep(args.update_freq * 60)
    except KeyboardInterrupt:
        print_status('interrupted, cleaning up...')
        try:
            if os.path.isfile(args.save_path):
                os.remove(args.save_path)
            os._exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
