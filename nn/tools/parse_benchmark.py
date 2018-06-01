#!/usr/bin/python3
""" NNAPI benchmark output parser (statistics aggregator)

Reads a file with output from multiple runs of
  adb shell am instrument
    -w com.example.android.nn.benchmark/android.support.test.runner.AndroidJUnitRunner
and provides aggregate statistics for benchmarks

TODO(mikie): provide json output for use with benchmarking

"""

import statistics
import sys

def main():
  stats = dict()
  with open(sys.argv[1]) as f:
    for line in f.readlines():
      if "INSTRUMENTATION_STATUS:" in line and "avg" in line:
        stat = line.split(": ")[1]
        name, value = stat.split("=")
        stats[name] = stats.get(name, []) + [float(value)]
  print("{0:<34}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}".format(
          "Benchmark", "mean", "stddev", "min", "max", "n"))
  for name in sorted(stats):
    values = stats[name]
    print("{0:<34}{1:>10.2f}{2:>10.2f}{3:>10.2f}{4:>10.2f}{5:>10d}".format(
            name, statistics.mean(values), statistics.stdev(values),
            min(values), max(values), len(values)))

if __name__ == '__main__':
  main()
