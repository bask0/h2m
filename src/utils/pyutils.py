"""
Python utils, reusable chunks etc.
"""

from typing import List, Union
import sys
import os
import time
import numpy as np
import logging


def rprint(value):
    """Similar to print function but overwrites last line.

    Parameters
    ----------
    value
        Value to print.

    """

    sys.stdout.write(f'\r{value}')
    sys.stdout.flush()


def print_progress(i: int, n_total: int, prefix: str = '') -> None:
    """Print progress bar.

    E.g. with ``prefix`` = 'training':

    training:  97% ||||||||||||||||||||||||||||||||||||||||||||||||   |

    Parameters
    ----------
    i
        Current step.
    n_total
        Total number of steps.
    prefix
        Printed in front of progress bar, limited to 20 characters.

    """
    perc = np.floor((i + 1) / n_total * 100)
    n_print = 50

    n_done = int(np.floor(perc / 100 * n_print))
    n_to_go = n_print - n_done

    if perc != 100:
        n_to_go = n_to_go-1
        msg = f'{perc:3.0f}% |{"|"*n_done}>{" "*n_to_go}' + '|'
    else:
        msg = f'{perc:3.0f}% |{"|"*n_done}{" "*n_to_go}' + '|'

    rprint(f'{prefix:20s} ' + msg)
    if perc == 100:
        print('\n')

class ProgressBar(object):
    def __init__(self, n):
        self.i = 0
        self.n = n

        self.t_start = time.time()
        
        self.print()
    
    def step(self):
        self.i += 1

        if self.i <= self.n:
            self.print()
        if self.i == self.n:
            print('\n')
        elif self.i > self.n:
            pass

    def print(self):
        perc = np.floor(self.i / self.n * 100)
        elapsed = time.time() - self.t_start
        remaining = np.ceil((elapsed / max(1, self.i)) * (self.n - self.i))
        rprint(f'{self.i:6} of {self.n} -- {perc:3.0f}% -- elapsed: {np.floor(elapsed):5.0f} s -- remaining: {remaining:5.0f} s')

def exit_if_exists(file: str, overwrite: bool = False):
    """Stop script if ``file`` exists and ``overwrite``is False.

    Parameters
    ----------
    file
        File path.
    overwrite (default: False)
        Whether to overwrite existing file (triggers warning) or not (quits if
        exists).

    """
    if os.path.exists(file):
        if overwrite:
            logging.warn('Overwriting existing file as overwrite is True.')
            os.remove(file)
        else:
            logging.info(
                'File exists and will not be overwritten as overwrite is False.')
            sys.exit(0)


def rm_existing(files: Union[str, List[str]]):
    """Remove file(s) if existing.

    Parapeters
    ----------
    files
        File(s) to remove.

    """
    if type(files) == str:
        files = [files]
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s - %s." % (e.f, e.strerror))
