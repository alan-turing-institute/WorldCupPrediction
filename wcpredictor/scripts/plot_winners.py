#!/usr/bin/env python

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser("plot the results of WC simulations")
    parser.add_argument("--input_csv", help="location of input file", required=True)
    parser.add_argument("--output_png", help="where to save image")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    df.sort_values(by="W", axis=0, ascending=False, inplace=True)
    # plot the top ten
    df = df[:10]
    fig, ax = plt.subplots(tight_layout=True)
    xvals = list(df.team.values)
    yvals = list(df.W.values)
    ax.bar(xvals, yvals)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax.set_ylabel("Number of wins")
    if args.output_png:
        plt.savefig(args.output_png)
    else:
        plt.show()



if __name__ == "__main__":
    main()
