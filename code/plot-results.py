#! /usr/bin/python
# -*- coding: utf-8 -*-

# imports
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy


def parse_arguments():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-v",
                    "--verbosity",
                    dest="verbosity",
                    choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                    default='ERROR',
                    help="Verbosity/Log level. Defaults to ERROR")
    parser.add_argument("-l",
                    "--logfile",
                    dest="logfile",
                    help="Store log to this file.")
    parser.add_argument("--result",
                    dest="results",
                    required=True,
                    help="Path to the results tsv file.")
    parser.add_argument("--plot-dir",
                    dest="plot_dir",
                    required=True,
                    help="Path for the plot directory")
    return parser


def facet_heatmap(data, val_col, **kws):
    '''Create n heatmaps on one plot'''
    data = data.pivot(index="t_size", columns='mean_1', values=val_col)
    data = data.reindex_axis(sorted(data.index)[::-1], axis=0)
    sns.heatmap(data, **kws)


def PLOT_cx_k_mutpb(df,cx,k,mut_pb,prefix,plot_dir,val_col):
    df = df.pivot(index="t_size", columns="mean_1", values=val_col)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    df = df.reindex_axis(sorted(df.index)[::-1], axis=0)
    ax = sns.heatmap(df, cmap="hot", annot=True, vmin=1.80, vmax=1.90, fmt=".3f", square=True)
    fig = ax.get_figure()
    fig.suptitle("Crossing-over:"+str(cx)[2:]+" Neighbours:"+str(k)+" Mutation Probability:"+str(mut_pb))
    fname = prefix+"_"+str(cx)+"_k"+str(k)+"_mutpb"+str(mut_pb)+".png"
    fig.savefig(os.path.join(os.path.join(plot_dir,"cx_k_mutpb"),fname),dpi=300)
    plt.close()


def PLOT_cx_mutpb(df,cx,mut_pb,prefix,plot_dir,val_col):
    g = sns.FacetGrid(df, col="k", col_wrap=2, size=3, aspect=2)
    g = g.map_dataframe(facet_heatmap,
    val_col=val_col, cmap="hot", vmin=1.80, vmax=1.90, annot=True, annot_kws={"size": 8}, fmt=".3f")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Crossing-over:"+str(cx)[2:]+" Mutation Probability:"+str(mut_pb))
    g.axes[0].set_ylabel("t_size")
    g.axes[2].set_ylabel("t_size")
    g.axes[2].set_xlabel("mean_1")
    g.axes[3].set_xlabel("mean_1")
    facecolor=plt.gcf().get_facecolor()
    for ax in g.axes.flat:
        ax.set_aspect('equal','box-forced')
        ax.set_axis_bgcolor(facecolor)
    fname = prefix+"_"+str(cx)+"_mutpb"+str(mut_pb)+".png"
    g.savefig(os.path.join(os.path.join(plot_dir,"cx_mutpb"),fname),dpi=300)
    plt.close()


def PLOT_k_mutpb(df,k,mut_pb,prefix,plot_dir,val_col):
    g = sns.FacetGrid(df, col="cx", col_wrap=3, size=3, aspect=2)
    g = g.map_dataframe(facet_heatmap,
    val_col=val_col, cmap="hot", vmin=1.80, vmax=1.90, annot=True, annot_kws={"size": 8}, fmt=".3f")
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle("Neighbours:"+str(k)+" Mutation Probability:"+str(mut_pb))
    g.axes[0].set_ylabel("t_size")
    facecolor=plt.gcf().get_facecolor()
    for ax in g.axes.flat:
        ax.set_aspect('equal','box-forced')
        ax.set_axis_bgcolor(facecolor)
        ax.set_xlabel("mean_1")
    fname = prefix+"_k"+str(k)+"_mutpb"+str(mut_pb)+".png"
    g.savefig(os.path.join(os.path.join(plot_dir,"k_mutpb"),fname),dpi=300)
    plt.close()


def PLOT_cx_k(df,cx,k,prefix,plot_dir,val_col):
    g = sns.FacetGrid(df, col="mut_pb", col_wrap=3, size=3, aspect=2)
    g = g.map_dataframe(facet_heatmap,
    val_col=val_col, cmap="hot", vmin=1.80, vmax=1.90, annot=True, annot_kws={"size": 8}, fmt=".3f")
    plt.subplots_adjust(top=0.8, hspace=0.4)
    g.fig.suptitle("Crossing-over:"+str(cx)[2:]+" Neighbours:"+str(k))
    g.axes[0].set_ylabel("t_size")
    facecolor=plt.gcf().get_facecolor()
    for ax in g.axes.flat:
        ax.set_aspect('equal','box-forced')
        ax.set_axis_bgcolor(facecolor)
        ax.set_xlabel("mean_1")
    fname = prefix+"_"+str(cx)+"_k"+str(k)+".png"
    g.savefig(os.path.join(os.path.join(plot_dir,"cx_k"),fname),dpi=300)

    plt.close()


if __name__ == '__main__':
    plot_dir = os.path.join('..', 'ga_res', 'qda_results')
    results = os.path.join(plot_dir, 'qda_results.tsv')

    def create_dir_if_not_exists(directory):
        try:
            os.mkdir(directory)
        except OSError:
            pass
    create_dir_if_not_exists(os.path.join(plot_dir, "cx_k_mutpb"))
    create_dir_if_not_exists(os.path.join(plot_dir,"cx_mutpb"))
    create_dir_if_not_exists(os.path.join(plot_dir,"k_mutpb"))
    create_dir_if_not_exists(os.path.join(plot_dir,"cx_k"))

    main_df = pd.read_csv(results, sep="\t", index_col=0)
    col_names = main_df.columns.values
    params = ["rep", "mean_1", "std_1", "cx", "mut_pb", "t_size", "elitism", "pop_size", "k", "max_t"]

    # some parameters are fixed, remove them, convert some floats to int
    # for i in params: print(i,list(set(main_df[i])))
    del main_df["std_1"]
    del main_df["elitism"]
    del main_df["pop_size"]
    del main_df["max_t"]

    # DataFrames for Fitness values for fixed params over all repetitions
    new_columns = list(copy.deepcopy(main_df.columns.values))
    new_columns.remove("rep")
    new_columns.remove("val")
    new_columns.remove("vector")
    avg_df = pd.DataFrame(columns=new_columns + ["avg_val", "std_val"])
    max_df = pd.DataFrame(columns=new_columns + ["max_val"])

    # Collapse over repetitions
    rep_groups = main_df.groupby(["mean_1", "cx", "mut_pb", "t_size", "k"])
    for name, group in rep_groups:
        mean_val = np.mean(group["val"])
        std_val = np.std(group["val"])
        max_val = max(group["val"])
        avg_df.loc[len(avg_df)] = (name[0], name[1], name[2], name[3], name[4], mean_val, std_val)
        max_df.loc[len(max_df)] = (name[0], name[1], name[2], name[3], name[4], max_val)
    avg_df["mean_1"] = avg_df["mean_1"].astype(int)
    avg_df["t_size"] = avg_df["t_size"].astype(int)
    avg_df["k"] = avg_df["k"].astype(int)
    avg_df.to_csv(os.path.join(plot_dir, "avg_df.tsv"), sep="\t", index=False)
    max_df["mean_1"] = max_df["mean_1"].astype(int)
    max_df["t_size"] = max_df["t_size"].astype(int)
    max_df["k"] = max_df["k"].astype(int)
    max_df.to_csv(os.path.join(plot_dir, "max_df.tsv"), sep="\t", index=False)

    # Find the fittest individuals
    max_fit = max([row["max_val"] for i, row in max_df.iterrows()])
    with open(os.path.join(plot_dir, "ga_knn_fittest.txt"), "w") as f:
        f.write(str(max_fit) + "\n")
        for i, row in main_df.iterrows():
            if row["val"] == max_fit:
                f.write(row["vector"] + "\n")

    # Plots for MAX df
    for cx in list(set(main_df["cx"])):
        cx_df = max_df[max_df["cx"] == cx]
        for k in list(set(main_df["k"])):
            cx_k_df = cx_df[cx_df["k"] == k]
            for mut_pb in list(set(main_df["mut_pb"])):
                cx_k_mutpb_df = cx_k_df[cx_k_df["mut_pb"] == mut_pb]
                PLOT_cx_k_mutpb(cx_k_mutpb_df, cx, k, mut_pb, "MAXF", plot_dir, "max_val")
    for cx in list(set(main_df["cx"])):
        cx_df = max_df[max_df["cx"] == cx]
        for mut_pb in list(set(main_df["mut_pb"])):
            cx_mutpb_df = cx_df[cx_df["mut_pb"] == mut_pb]
            PLOT_cx_mutpb(cx_mutpb_df, cx, mut_pb, "MAXF", plot_dir, "max_val")
    for k in list(set(main_df["k"])):
        k_df = max_df[max_df["k"] == k]
        for mut_pb in list(set(main_df["mut_pb"])):
            k_mutpb_df = k_df[k_df["mut_pb"] == mut_pb]
            PLOT_k_mutpb(k_mutpb_df, k, mut_pb, "MAXF", plot_dir, "max_val")
    for cx in list(set(main_df["cx"])):
        cx_df = max_df[max_df["cx"] == cx]
        for k in list(set(main_df["k"])):
            cx_k_df = cx_df[cx_df["k"] == k]
            PLOT_cx_k(cx_k_df, cx, k, "MAXF", plot_dir, "max_val")

    # Plots for AVG df
    for cx in list(set(main_df["cx"])):
        cx_df = avg_df[avg_df["cx"] == cx]
        for k in list(set(main_df["k"])):
            cx_k_df = cx_df[cx_df["k"] == k]
            for mut_pb in list(set(main_df["mut_pb"])):
                cx_k_mutpb_df = cx_k_df[cx_k_df["mut_pb"] == mut_pb]
                PLOT_cx_k_mutpb(cx_k_mutpb_df, cx, k, mut_pb, "AVGF", plot_dir, "avg_val")
    for cx in list(set(main_df["cx"])):
        cx_df = avg_df[avg_df["cx"] == cx]
        for mut_pb in list(set(main_df["mut_pb"])):
            cx_mutpb_df = cx_df[cx_df["mut_pb"] == mut_pb]
            PLOT_cx_mutpb(cx_mutpb_df, cx, mut_pb, "AVGF", plot_dir, "avg_val")
    for k in list(set(main_df["k"])):
        k_df = avg_df[avg_df["k"] == k]
        for mut_pb in list(set(main_df["mut_pb"])):
            k_mutpb_df = k_df[k_df["mut_pb"] == mut_pb]
            PLOT_k_mutpb(k_mutpb_df, k, mut_pb, "AVGF", plot_dir, "avg_val")
    for cx in list(set(main_df["cx"])):
        cx_df = avg_df[avg_df["cx"] == cx]
        for k in list(set(main_df["k"])):
            cx_k_df = cx_df[cx_df["k"] == k]
            PLOT_cx_k(cx_k_df, cx, k, "AVGF", plot_dir, "avg_val")
