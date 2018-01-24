#! /usr/bin/python
# -*- coding: utf-8 -*-

# imports
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy

def PLOT_cx_mutpb(df, cx, mut_pb, prefix, plot_dir, val_col):
    df = df.pivot(index="t_size", columns="mean_1", values=val_col)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    df = df.reindex_axis(sorted(df.index)[::-1], axis=0)
    ax = sns.heatmap(df, cmap="hot", annot=True, vmin=1.90, vmax=1.96, fmt=".3f", square=True)
    fig = ax.get_figure()
    fig.suptitle("Crossing-over:" + str(cx)[2:] + " Mutation Probability:" + str(mut_pb))
    fname = prefix + "_" + str(cx) + "_mutpb" + str(mut_pb) + ".png"
    fig.savefig(os.path.join(os.path.join(plot_dir, "cx_mutpb"), fname), dpi=300)
    plt.close()



if __name__ == '__main__':
    plot_dir = os.path.join('..', 'ga_res', 'qda_results')
    results = os.path.join(plot_dir, 'qda_results.tsv')


    def create_dir_if_not_exists(directory):
        try:
            os.mkdir(directory)
        except OSError:
            pass


    create_dir_if_not_exists(os.path.join(plot_dir, "cx_mutpb"))

    main_df = pd.read_csv(results, sep="\t", index_col=0)
    col_names = main_df.columns.values
    params = ["rep", "mean_1", "std_1", "cx", "mut_pb", "t_size", "elitism", "pop_size", "max_t"]

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
    rep_groups = main_df.groupby(["mean_1", "cx", "mut_pb", "t_size"])
    for name, group in rep_groups:
        mean_val = np.mean(group["val"])
        std_val = np.std(group["val"])
        max_val = max(group["val"])
        avg_df.loc[len(avg_df)] = (name[0], name[1], name[2], name[3], mean_val, std_val)
        max_df.loc[len(max_df)] = (name[0], name[1], name[2], name[3], max_val)
    avg_df["mean_1"] = avg_df["mean_1"].astype(int)
    avg_df["t_size"] = avg_df["t_size"].astype(int)
    avg_df.to_csv(os.path.join(plot_dir, "avg_df.tsv"), sep="\t", index=False)
    max_df["mean_1"] = max_df["mean_1"].astype(int)
    max_df["t_size"] = max_df["t_size"].astype(int)
    max_df.to_csv(os.path.join(plot_dir, "max_df.tsv"), sep="\t", index=False)

    # Find the fittest individuals
    max_fit = max([row["max_val"] for i, row in max_df.iterrows()])
    with open(os.path.join(plot_dir, "ga_qda_fittest.txt"), "w") as f:
        f.write(str(max_fit) + "\n")
        for i, row in main_df.iterrows():
            if row["val"] == max_fit:
                f.write(row["vector"] + "\n")

    # Plots for MAX df
    for cx in list(set(main_df["cx"])):
        cx_df = max_df[max_df["cx"] == cx]
        for mut_pb in list(set(main_df["mut_pb"])):
            cx_mutpb_df = cx_df[cx_df["mut_pb"] == mut_pb]
            PLOT_cx_mutpb(cx_mutpb_df, cx, mut_pb, "MAXF", plot_dir, "max_val")

    # Plots for AVG df
    for cx in list(set(main_df["cx"])):
        cx_df = avg_df[max_df["cx"] == cx]
        for mut_pb in list(set(main_df["mut_pb"])):
            cx_mutpb_df = cx_df[cx_df["mut_pb"] == mut_pb]
            PLOT_cx_mutpb(cx_mutpb_df, cx, mut_pb, "AVGF", plot_dir, "avg_val")
