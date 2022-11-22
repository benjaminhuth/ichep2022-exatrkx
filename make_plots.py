import sys
import itertools

from pathlib import Path
from datetime import datetime

import torch
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_performance_dataframes(rootfile, min_pT=0, max_pT=100, length_cut=0):
    assert 0 <= min_pT and min_pT < max_pT
    assert length_cut >= 0

    performance_particles = uproot.open("{}:track_finder_particles".format(rootfile))
    performance_tracks = uproot.open("{}:track_finder_tracks".format(rootfile))

    # Particle based metrics
    particles_df = pd.DataFrame()
    for key in performance_particles.keys():
        particles_df[key] = performance_particles[key].array(library="np")

    particles_df["pT"] = np.hypot(particles_df.px, particles_df.py)
    particles_df["eta"] = np.arctanh(particles_df.pz/np.hypot(particles_df.pT, particles_df.pz))
    particles_df["theta"] = np.degrees(np.arctan(np.hypot(particles_df.pT, particles_df.pz)/particles_df.pz))
    particles_df = particles_df[ particles_df["pT"] < max_pT ]
    particles_df = particles_df[ particles_df["pT"] > min_pT ]
    particles_df = particles_df[ particles_df["nhits"] > length_cut ]

    # Track based metrics
    tracks_df = pd.DataFrame()

    for key in performance_tracks.keys():
        if key not in ['particle_id', 'particle_nhits_total', 'particle_nhits_on_track']:
            tracks_df[key] = performance_tracks[key].array(library="np")
        else:
            arrays = performance_tracks[key].array(library="np")
            tracks_df[key] = [ a[0] for a in arrays ]

    tracks_df = tracks_df.rename(columns={
        "particle_id": "maj_particle_id",
        "particle_nhits_total": "maj_particle_nhits_total",
        "particle_nhits_on_track": "maj_particle_nhits_on_track",
    })

    tracks_df["purity"] = tracks_df["maj_particle_nhits_on_track"] / tracks_df["size"]
    tracks_df["efficiency"] = tracks_df["maj_particle_nhits_on_track"] / tracks_df["maj_particle_nhits_total"]
    tracks_df = tracks_df[ tracks_df["size"] > length_cut ]

    pT = np.hypot(particles_df["px"].to_numpy(), particles_df["py"].to_numpy())
    p = np.hypot(pT, particles_df["pz"].to_numpy())
    pL = particles_df["pz"].to_numpy()

    tracks_df["pT"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], pT)))
    tracks_df["eta"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], np.arctanh(pL/p))))
    tracks_df["theta"] = tracks_df["maj_particle_id"].map(dict(zip(particles_df["particle_id"], np.degrees(np.arctan(p/pL)))))

    # Efficiencies for particles
    f = lambda x: x.maj_particle_nhits_on_track.max() / x.maj_particle_nhits_total[x.index[0]]
    efficiency = tracks_df.groupby(["maj_particle_id", "event_id"]).apply(f).reset_index()
    efficiency_dict = dict(zip(list(efficiency[["maj_particle_id", "event_id"]].itertuples(index=False, name=None)), efficiency[0]))
    particles_df["efficiencies"] = particles_df.set_index(['particle_id', 'event_id']).index.map(efficiency_dict).fillna(0)

    # Efficiency-thresholds for particles
    for threshold in [0.5, 0.75, 0.9, 0.999]:
        reconstructed = tracks_df[ tracks_df["efficiency"] > threshold ][["maj_particle_id","event_id"]]

        particles_multi_index = particles_df.set_index(["particle_id","event_id"]).index
        reconstructed_multi_index = reconstructed.set_index(["maj_particle_id","event_id"]).index

        particles_df["reconstructed_{}".format(int(threshold*100))] = particles_multi_index.isin(reconstructed_multi_index).astype(int)

    return particles_df, tracks_df


def plot_binned_2d(ax, x, y, bins, threshold=10, do_scatter=True, **plot_kwargs):
    hist, edges, _ = np.histogram2d(x, y, (bins,2))
    mask = np.sum(hist, axis=1) > threshold
    plotpoints = (edges[:-1] + np.diff(edges))[mask], hist[:,1][mask] / np.sum(hist, axis=1)[mask]
    line_plots = ax.plot(*plotpoints, **plot_kwargs)
    if do_scatter:
        ax.scatter(*plotpoints, color=line_plots[0]._color)

    return ax


def make_efficiency_plots(particles_df):
    fig, ax = plt.subplots(1,3,figsize=(22,6))
    fig.suptitle("Fraction of reconstructed particles with different efficiency thresholds")

    ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_50, 25, label="50%")
    ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_75, 25, label="75%")
    ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_90, 25, label="90%")
    ax[0] = plot_binned_2d(ax[0], particles_df.eta, particles_df.reconstructed_99, 25, label="100%")
    ax[0].set_ylim(0,1)
    ax[0].set_xlabel("$\\eta$")
    ax[0].set_ylabel("fraction of reconstructed particles")
    ax[0].legend()
    ax[1] = plot_binned_2d(ax[1], particles_df.nhits, particles_df.reconstructed_50, 20, label="50%")
    ax[1] = plot_binned_2d(ax[1], particles_df.nhits, particles_df.reconstructed_75, 20, label="75%")
    ax[1] = plot_binned_2d(ax[1], particles_df.nhits, particles_df.reconstructed_90, 20, label="90%")
    ax[1] = plot_binned_2d(ax[1], particles_df.nhits, particles_df.reconstructed_99, 20, label="100%")
    ax[1].set_xlabel("track length [# hits]")
    ax[1].set_ylim(0,1)
    ax[2] = plot_binned_2d(ax[2], particles_df.pT, particles_df.reconstructed_50, 20, label="50%")
    ax[2] = plot_binned_2d(ax[2], particles_df.pT, particles_df.reconstructed_75, 20, label="75%")
    ax[2] = plot_binned_2d(ax[2], particles_df.pT, particles_df.reconstructed_90, 20, label="90%")
    ax[2] = plot_binned_2d(ax[2], particles_df.pT, particles_df.reconstructed_99, 20, label="100%")
    ax[2].set_xlabel("pT [GeV]")
    _ = ax[2].set_ylim(0,1)

def make_time_comparison_plot(tsv_file):
    timing = pd.read_csv(tsv_file, sep='\t')

    fig, ax = plt.subplots()


    def trim_label(name):
        if name[:4] == "Algo":
            name = name[10:]
        if name[-4:] == "ithm":
            name = name[:-9]
        return name

    label_conversions = {
        "TrackFindingMLBased" : "Exa.TrkX",
        "TrackFinding" : "Combinatorial Kalman Filter",
        "TrackParamsEstimation" : "Parameter Estimation",
        "TrackFitting" : "Kalman Filter",
        "Seeding" : "Seeding",
    }
    def filter_label(label):
        if label in label_conversions.keys():
            return label_conversions[label]
        else:
            return None

    unique_labels = []
    def only_unique(label):
        if label in unique_labels:
            return None
        else:
            unique_labels.append(label)
            return label

    def make_label(label):
        return only_unique(filter_label(trim_label(label)))

    bottom = 0
    for i in [5, 6, 7]:
        time = timing.iloc[i][2]
        l = make_label(timing.iloc[i][0])
        ax.bar('Exa.TrkX', time, label=l, bottom=bottom)
        bottom += time

    bottom = 0
    for i, c in zip([14, 16], ['tab:red', 'tab:cyan']):
        time = timing.iloc[i][2]
        l = make_label(timing.iloc[i][0])
        ax.bar('CKF', time, label=l, bottom=bottom, color=c)
        bottom += time

    bottom = 0
    for i, c in zip([8, 9, 11], ['tab:purple', 'tab:orange', 'tab:cyan']):
        time = timing.iloc[i][2]
        l = make_label(timing.iloc[i][0])
        ax.bar('Truth CKF', time, label=l, bottom=bottom, color=c)
        bottom += time

    bottom = 0
    for i, c in zip([8, 9, 10], ['tab:brown', 'tab:orange', 'tab:green']):
        time = timing.iloc[i][2]
        l = make_label(timing.iloc[i][0])
        ax.bar('Truth Tracking', time, label=l, bottom=bottom, color=c)
        bottom += time


    #plt.ylim(0, 2.5)

    # ax.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax.legend(loc="upper right")


    ax.set_ylabel("time [s]")
    ax.set_title("Time comparison")

    return fig

def make_gpu_memory_plot(csv_file, gpu_id):
    profile = pd.read_csv(csv_file)
    profile = profile[ profile[" index"] == gpu_id ]

    timestamps = profile["timestamp"].tolist()
    timestamps = [ datetime.strptime(tp, '%Y/%m/%d %H:%M:%S.%f') for tp in timestamps ]

    memory_gb = profile[" memory.used [MiB]"].to_numpy() / 1000

    # Start quickly before we reach a threshold
    start = np.nonzero(memory_gb > 0.1)[0][0]
    start = max(start-100, 0)

    memory_gb = memory_gb[start:]
    timestamps = timestamps[start:]

    time_s = [ (tp - timestamps[0]).total_seconds() for tp in timestamps ]

    fig, ax = plt.subplots()
    ax.plot(time_s, memory_gb, label="Exa.TrkX full chain")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("GPU memory usage [GB]")
    ax.set_title("GPU memory consumption for 10 events")
    ax.legend(loc='lower right')

    return fig


def cantor_pairing(a):
    return a[1] + ((a[0] + a[1])*(a[0] + a[1] + 1))//2

def eff_pur_cantor(pred, true):
    pred = np.sort(pred, axis=0)
    true = np.sort(true, axis=0)

    cantor_true = cantor_pairing(true)
    cantor_pred = cantor_pairing(pred)

    # print(len(cantor_true))
    # print(len(cantor_pred))

    cantor_intersection = np.intersect1d(cantor_pred, cantor_true)

    return (
        len(cantor_intersection)/len(cantor_true), # eff = true_in_pred / all_true
        len(cantor_intersection)/len(cantor_pred)  # pur = true_in_pred / all_pred
    )


def plot_eff_pur(ax, base_dir, datatype, event_number_str):
    emb_path = base_dir / "embedding_output" / datatype / event_number_str
    # print(emb_path)
    emb = torch.load(emb_path, map_location=torch.device('cpu'))
    emb_eff, emb_pur = eff_pur_cantor(emb.edge_index.numpy(), emb.modulewise_true_edges.numpy())

    flt_path = base_dir / "filter_output" / datatype / event_number_str
    # print(flt_path)
    flt = torch.load(flt_path, map_location=torch.device('cpu'))
    flt_eff, flt_pur = eff_pur_cantor(flt.edge_index.numpy(), flt.modulewise_true_edges.numpy())

    gnn_path = base_dir / "gnn_output" / datatype / event_number_str
    # print(gnn_path)
    gnn = torch.load(gnn_path, map_location=torch.device('cpu'))
    gnn_idxs = gnn.scores.numpy()[:len(gnn.scores)//2] > 0.5
    gnn_eff, gnn_pur = eff_pur_cantor(gnn.edge_index.numpy().T[gnn_idxs].T, gnn.modulewise_true_edges.numpy())
    gnn_eff, gnn_pur

    pos = [0.2,1,1.8]

    ax[0].plot(pos, [emb_eff, flt_eff, gnn_eff])
    ax[0].scatter(pos, [emb_eff, flt_eff, gnn_eff])
    ax[1].plot(pos, [emb_pur, flt_pur, gnn_pur])
    ax[1].scatter(pos, [emb_pur, flt_pur, gnn_pur])

    for axx in ax:
        axx.set_xlim(0,2)
        axx.set_xticks(pos)
        axx.set_xticklabels(["graph building","graph filtering","GNN"])

def make_eff_pur_detector_map(positions, all_edges, true_edges, ax, prec=2):
    def edges_in_radius(x, edge_index, r_range, z_range):
        x_idxs = np.nonzero(np.logical_and.reduce((
            x[:,0] > r_range[0],
            x[:,0] <= r_range[1],
            x[:,2] > z_range[0],
            x[:,2] <= z_range[1])))

        edge_idxs = np.logical_and(
            np.isin(edge_index[0], x_idxs),
            np.isin(edge_index[1], x_idxs))

        return edge_index[:, edge_idxs]

    # Go over all parts of the detector
    r_ranges = []
    z_ranges = []

    # Pixel
    for z_range in [(-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0)]:
        r_ranges.append((0.0,0.2))
        z_ranges.append(z_range)

    # SStrip, LStrip
    for r_range in [(0.2,0.7), (0.7,1.2)]:
        for z_range in [(-3.1, -1.2), (-1.2,1.2), (1.2, 3.1)]:
            r_ranges.append(r_range)
            z_ranges.append(z_range)

    colors = ['r', 'g', 'b', 'y', 'c']
    for r_range, z_range, color in zip(r_ranges, z_ranges, itertools.cycle(colors)):
        all_edges_selected = edges_in_radius(positions, all_edges, r_range, z_range)
        true_edges_selected = edges_in_radius(positions, true_edges, r_range, z_range)

        # print(r_range, z_range)

        eff, pur = eff_pur_cantor(all_edges_selected, true_edges_selected)
        fmt_str = "eff: {:." + str(prec) + "f} \npur: {:." + str(prec) + "f}"
        ax.text(
            np.mean(z_range)*1.1 - 0.5,
            np.mean(r_range) - 0.05,
            fmt_str.format(eff, pur))
        rectangle_args = {
            "xy": (z_range[0], r_range[0]),
            "width": (z_range[1]-z_range[0]),
            "height": (r_range[1]-r_range[0]),
        }
        ax.add_patch(plt.Rectangle(**rectangle_args, alpha=0.1, color=color, linewidth=0))


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    show = True

    #####################
    # Inference results #
    #####################

    inference_result_dir_smear = Path("inference_results_smeared/with_selection")
    inference_result_dir_truth = Path("inference_results_truth/with_selection")

    performance_cuts = {
        "max_pT": 100,
        "min_pT": 0.5,
        "length_cut": 8,
    }

    if False:
        fig = make_gpu_memory_plot(inference_result_dir_smear / "gpu_memory_profile.csv", gpu_id=3)
        fig.savefig("memory_profile.pdf",bbox_inches='tight', pad_inches = 0)

    if False:
        fig = make_time_comparison_plot(inference_result_dir_smear / "timing.tsv")
        fig.savefig("timing.pdf",bbox_inches='tight', pad_inches = 0)

    if False:
        particles_df_exa_smear, tracks_df_exa_smear = make_performance_dataframes(
            inference_result_dir_smear / "track_finding_performance_exatrkx.root",
            **performance_cuts)
        particles_df_exa_truth, tracks_df_exa_truth = make_performance_dataframes(
            inference_result_dir_truth / "track_finding_performance_exatrkx.root",
            **performance_cuts)

        fig, ax = plt.subplots()
        ax = plot_binned_2d(ax, particles_df_exa_smear.eta, particles_df_exa_smear.reconstructed_90,
                            25, do_scatter=False, label="smeared hits (90%)", color="tab:blue")
        ax = plot_binned_2d(ax, particles_df_exa_smear.eta, particles_df_exa_smear.reconstructed_50,
                            25, do_scatter=False, label="smeared hits (50%)", color="tab:blue", ls="--")
        ax = plot_binned_2d(ax, particles_df_exa_truth.eta, particles_df_exa_truth.reconstructed_90,
                            25, do_scatter=False, label="true hits (90%)", color="tab:orange")
        ax = plot_binned_2d(ax, particles_df_exa_truth.eta, particles_df_exa_truth.reconstructed_50,
                            25, do_scatter=False, label="true hits (50%)", color="tab:orange", ls="--")
        ax.legend()
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("fraction of particles reconstructed")
        ax.set_title("Reconstruction efficiencies for Exa.TrkX")
        ax.set_ylim(0,1)
        fig.tight_layout()
        fig.savefig("efficiency_exatrkx_smeared_vs_truth.pdf",bbox_inches='tight', pad_inches = 0)

    if False:
        particles_df_ckf_smear, tracks_df_ckf_smear = make_performance_dataframes(
            inference_result_dir_smear / "track_finding_performance_ckf.root",
            **performance_cuts)
        particles_df_ckf_truth, tracks_df_ckf_truth = make_performance_dataframes(
            inference_result_dir_truth / "track_finding_performance_ckf.root",
            **performance_cuts)

        fig, ax = plt.subplots()
        ax = plot_binned_2d(ax, particles_df_ckf_smear.eta, particles_df_ckf_smear.reconstructed_90,
                            25, do_scatter=False, label="smeared hits (90%)", color="tab:blue")
        ax = plot_binned_2d(ax, particles_df_ckf_smear.eta, particles_df_ckf_smear.reconstructed_50,
                            25, do_scatter=False, label="smeared hits (50%)", color="tab:blue", ls="--")
        ax = plot_binned_2d(ax, particles_df_ckf_truth.eta, particles_df_ckf_truth.reconstructed_90,
                            25, do_scatter=False, label="true hits (90%)", color="tab:orange")
        ax = plot_binned_2d(ax, particles_df_ckf_truth.eta, particles_df_ckf_truth.reconstructed_50,
                            25, do_scatter=False, label="true hits (50%)", color="tab:orange", ls="--")
        ax.legend()
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("fraction of particles reconstructed")
        ax.set_title("Reconstruction efficiencies for the CKF")
        ax.set_ylim(0,1)
        fig.tight_layout()
        fig.savefig("efficiency_ckf_smeared_vs_truth.pdf",bbox_inches='tight', pad_inches = 0)


    ####################
    # Training results #
    ####################

    training_artifact_dir_truth = Path("inference_results_truth/tmp")
    training_artifact_dir_smear = Path("inference_results_smeared/tmp")

    if False:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax = [ax1, ax2]
        plot_eff_pur(ax, training_artifact_dir_smear, "train", "0000")
        plot_eff_pur(ax, training_artifact_dir_truth, "train", "0000")

        ax[0].legend(["smeared hits", "_smear", "truth hits", "_truth"])
        ax[1].legend(["smeared hits", "_smear", "truth hits", "_truth"])

        ax[0].set_title("Training Efficiency")
        ax[1].set_title("Training Purity")

        ax[0].set_ylabel("efficiency")
        ax[1].set_ylabel("purity")

        fig1.savefig("training_eff.pdf")
        fig2.savefig("training_pur.pdf")
        # plt.show()

    if True:
        def make_detector_plot(event_file, **kwargs):
            fig, ax = plt.subplots()

            try:
                idxs = (event_file.scores[:len(event_file.scores)//2] > 0.5).numpy()
                edge_index = event_file.edge_index[:, idxs].numpy()
            except:
                edge_index = event_file.edge_index.numpy()


            # draw less dots to reduce file size
            x_to_draw = event_file.x.numpy().copy()
            np.random.shuffle(x_to_draw)
            x_to_draw = x_to_draw[:len(event_file.x)//10]

            ax.scatter(x_to_draw[:,2], x_to_draw[:,0], s=1, color='lightgrey')
            ax.set_xlabel("z [m]")
            ax.set_ylabel("r [m]")
            make_eff_pur_detector_map(event_file.x.numpy(),
                                    edge_index,
                                    event_file.modulewise_true_edges.numpy(),
                                    ax, **kwargs)

            return fig, ax

        # Truth
        gnn_truth = torch.load(training_artifact_dir_truth / "gnn_output" / "train" / "0000", map_location='cpu')
        fig, ax = make_detector_plot(gnn_truth)
        ax.set_title("Metrics for truth training")
        fig.savefig("detector_metrics_truth.pdf")

        gnn_smeared = torch.load(training_artifact_dir_smear / "gnn_output" / "train" / "0000", map_location='cpu')
        fig, ax = make_detector_plot(gnn_smeared)
        ax.set_title("Metrics for smeared training")
        fig.savefig("detector_metrics_smeared.pdf")

        embedding_smeared = torch.load(training_artifact_dir_truth / "embedding_output" / "train" / "0000", map_location='cpu')
        fig, ax = make_detector_plot(embedding_smeared, prec=4)
        ax.set_title("Embedding stage (smeared)")
        fig.savefig("detector_metrics_smeared_embedding.pdf")

        filter_smeared = torch.load(training_artifact_dir_truth / "filter_output" / "train" / "0000", map_location='cpu')
        fig, ax = make_detector_plot(filter_smeared)
        ax.set_title("Filter stage (smeared)")
        fig.savefig("detector_metrics_smeared_filter.pdf")

    if show:
        plt.show()




























