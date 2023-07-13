import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def correlate(
        x: np.ndarray,
        y: np.ndarray
) -> np.ndarray:
    sample_count = len(x)
    result = np.zeros(sample_count)

    for k in range(sample_count):
        r = 0.
        for n in range(sample_count):
            if n - k < 0:
                continue
            r += x[n] * y[n - k]
        result[k] = r

    result /= result[0]

    return result


def process_signals(
        x: np.ndarray,
        y: Optional[np.ndarray] = None
) -> np.ndarray:
    if y is None:
        result = correlate(x, x)
    else:
        if len(x) != len(y):
            raise ValueError('Signal lengths must match for correlation')
        result = correlate(x, y)

    return result


def read_signal(
        filename: str,
        channel: int
) -> np.ndarray:
    df = pd.read_csv(
        filename,
        dtype='float64',
        sep=';'
    )

    if channel >= df.shape[1]:
        raise ValueError(f'Channel {channel} does not exist')

    return df.iloc[:, channel]


def plot_signal(
        signal: np.ndarray,
        title: str
) -> None:
    plt.figure()
    plt.plot(signal)
    plt.title(title)
    plt.show()


def save_signal(
        filename: str,
        signal: np.ndarray
) -> None:
    df = pd.DataFrame(signal)
    df.to_csv(filename, index=False)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='This program reads signal(s) from a file, computes the '
                    'correlation/autocorrelation function based on the '
                    'formula for discrete signals from the lecture, plots the '
                    'signals and optionally saves the result to a file'
    )
    parser.add_argument(
        'signal',
        help='The name of the file to read the base signal from'
    )
    parser.add_argument(
        '--reference-signal',
        '-s',
        help='The name of the file to read the reference signal from. If this '
             'is provided the program will compute its correlation with the '
             'base signal'
    )
    parser.add_argument(
        '--out',
        '-o',
        help='The name of the file to save results to'
    )
    parser.add_argument(
        '--channel',
        '-c',
        help='The channel to use. Defaults to 0',
        type=int,
        default=0
    )
    return parser


def main():
    args = create_parser().parse_args()
    signal = args.signal
    reference_signal = args.reference_signal
    out = args.out
    channel = args.channel

    x = read_signal(signal, channel)
    r: np.ndarray

    if reference_signal is None:
        r = process_signals(x)
        plot_signal(x, 'Signal')
    else:
        y = read_signal(reference_signal, channel)
        r = process_signals(x, y)
        plot_signal(x, 'Signal 1')
        plot_signal(y, 'Signal 2')
    plot_signal(
        r,
        ('Autoc' if reference_signal is None else 'C') + 'orrelation'
    )

    if out is not None:
        save_signal(out, r)


if __name__ == '__main__':
    main()
