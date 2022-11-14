#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="plot how far teams got in the tournament")
    parser.add_argument("--input_csv",help="location of input file")
    parser.add_argument("--output_png",help="location of output file")
    parser.add_argument("--team_list",help="comma-separated list of teams", default="England,Wales")
    args = parser.parse_args()
    teams = args.team_list.split(",")
    title_string = ",".join(teams[:-1]) + f" and {teams[-1]}"
    # open the dataframe containing progress for each team
    df = pd.read_csv(args.input_csv)
    # x-axis labels
    labels = ['Group', 'R16', 'QF', 'SF', 'RU', 'W']
    # setup the plots
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    offset = width * (len(teams)-1)/ 2
    # y-axis values for each selected team
    for i,team in enumerate(teams):
        row = df.loc[df.team==team]
        stages = [
            row.G.values[0],
            row.R16.values[0],
            row.QF.values[0],
            row.SF.values[0],
            row.RU.values[0],
            row.W.values[0]
        ]
        rect = ax.bar(x - offset + i*width, stages, width, label=team)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of tournaments')
    ax.set_title(title_string)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    if args.output_png:
        plt.savefig(args.output_png)
    else:
        plt.show()


if __name__ == "__main__":
    main()
