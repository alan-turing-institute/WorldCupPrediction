#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sim_results.csv")

df.sort_values(by="W", axis=0, ascending=False, inplace=True)
df = df[:10]

df.plot.bar(x="team",y="W")

plt.show()
