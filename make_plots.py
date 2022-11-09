import sys

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_time_comparison_plot(tsv_file, show=False):
    timing = pd.read_csv(tsv_file, sep='\t')

    fig, ax = plt.subplots()

    def trim_name(name):
        if name[:4] == "Algo":
            name = name[10:]
        if name[-4:] == "ithm":
            name = name[:-9]
        return name

    bottom = 0
    for i in [5, 6, 7]:
        time = timing.iloc[i][2]
        l = trim_name(timing.iloc[i][0])
        ax.bar('Exa.TrkX', time, label="{} [{:.2f} s]".format(l, time), bottom=bottom)
        bottom += time

    bottom = 0
    for i, c in zip([14, 16], ['tab:red', 'tab:cyan']):
        time = timing.iloc[i][2]
        l = trim_name(timing.iloc[i][0])
        ax.bar('CKF', time, label="{} [{:.2f} s]".format(l, time), bottom=bottom, color=c)
        bottom += time

    bottom = 0
    for i, c in zip([8, 9, 11], ['tab:purple', 'tab:orange', 'tab:cyan']):
        time = timing.iloc[i][2]
        l = trim_name(timing.iloc[i][0])
        ax.bar('Truth CKF*', time, label="{} [{:.2f} s]".format(l, time), bottom=bottom, color=c)
        bottom += time

    bottom = 0
    for i, c in zip([8, 9, 10], ['tab:brown', 'tab:orange', 'tab:green']):
        time = timing.iloc[i][2]
        l = trim_name(timing.iloc[i][0])
        ax.bar('Truth Tracking**', time, label="{} [{:.2f} s]".format(l, time), bottom=bottom, color=c)
        bottom += time


    #plt.ylim(0, 2.5)

    ax.legend(bbox_to_anchor=(1,1), loc="upper left")

    ax.set_ylabel("time [s]")
    ax.set_title("Time comparison")

    if show:
        plt.show()

def make_gpu_memory_plot(csv_file, gpu_id, show=False):
    profile = pd.read_csv(csv_file)

    profile = profile[ profile["index"] == gpu_id ]
    plt.plot(profile.index)

    timestamps = profile["timestamp"].tolist()
    timestamps = [ datetime.datetime(tp) for tp in timestamps ]
    t = [ (tp - times[0]).total_seconds() for tp in timestamps ]

    plt.plot(t, profile["memory.used [MiB]"].to_numpy() / 1000 )
    plt.show()


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])

    make_gpu_memory_plot(data_dir / "gpu_memory_profile.csv", 3)



