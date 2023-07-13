```text
usage: correlation.py [-h] [--reference-signal REFERENCE_SIGNAL] [--out OUT]
                      [--channel CHANNEL]
                      signal

This program reads signal(s) from a file, computes the
correlation/autocorrelation function based on the formula for discrete signals
from the lecture, plots the signals and optionally saves the result to a file

positional arguments:
  signal                The name of the file to read the base signal from

options:
  -h, --help            show this help message and exit
  --reference-signal REFERENCE_SIGNAL, -s REFERENCE_SIGNAL
                        The name of the file to read the reference signal
                        from. If this is provided the program will compute its
                        correlation with the base signal
  --out OUT, -o OUT     The name of the file to save results to
  --channel CHANNEL, -c CHANNEL
                        The channel to use. Defaults to 0
```