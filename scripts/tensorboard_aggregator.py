import os
import sys
import argparse
import shutil
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(exp_path):

    seeds = [s for s in os.listdir(exp_path) if "combined" not in s]
    summary_iterators = [EventAccumulator(os.path.join(exp_path, dname)).Reload() for dname in seeds]

    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out


def create_histogram_summary(tag, values, bins=1000):
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    return tf.Summary.Value(tag=tag, histo=hist)


def create_parsed_histogram_summary(tag, values, bins=1000):
    # Convert to a numpy array

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    return tf.Summary.Value(tag=tag, histo=hist)


def write_combined_events(exp_path, d_combined, dname='combined', mean_var_tags=()):

    fpath = os.path.join(exp_path, dname)
    if os.path.isdir(fpath):
        shutil.rmtree(fpath)
    assert not os.path.isdir(fpath)

    writer = tf.summary.FileWriter(fpath)


    tags, values = zip(*d_combined.items())

    cap = min([len(v) for v in values])
    values = [v[:cap] for v in values]

    timestep_mean = np.array(values).mean(axis=-1)
    timestep_var = np.array(values).var(axis=-1)
    timesteps = timestep_mean[tags.index("frames")]

    for tag, means, vars in zip(tags, timestep_mean, timestep_var):
        for i, mean, var in zip(timesteps, means, vars):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=mean)])
            writer.add_summary(summary, global_step=i)
            writer.flush()

            if tag in mean_var_tags:
                values = np.array([mean - var, mean, mean + var])

                summary = tf.Summary(value=[
                    create_histogram_summary(tag=tag+"_var", values=values)
                ])
                writer.add_summary(summary, global_step=i)
                writer.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dpath = sys.argv[1]
    else:
        raise ValueError("Specify dir")

    parser = argparse.ArgumentParser()

    parser.add_argument('--experiments', nargs='+', help='experiment directories to aggregate', required=True)

    parser.add_argument('--mean-var-tags', nargs='+', help='tags to create mean-var histograms from', required=False, default=["return_mean"])

    args = parser.parse_args()

    for exp_path in args.experiments:
        d = tabulate_events(exp_path)
        write_combined_events(exp_path, d, mean_var_tags=args.mean_var_tags)