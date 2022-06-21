import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from collections import OrderedDict


class BarUniqueDtypeNum:
    def __init__(self, df, color="blue", edgecolor="k", linewidth=2, figsize=(8, 6)):
        self.df = df
        self.color = color
        self.edgecolor = edgecolor
        self.figsize = figsize
        self.linewidth = linewidth

    def __call__(self, dtypes):
        self.df.select_dtypes(dtypes).nunique().value_counts().sort_index().plot.bar(
            color=self.color,
            figsize=self.figsize,
            edgecolor=self.edgecolor,
            linewidth=self.linewidth,
        )
        plt.xlabel("Number of Unique Values")
        plt.ylabel("Count")
        plt.title("Count of Unique Values in Integer Columns")


class KdePlotByTarget:
    def __init__(self, df, figsize=(20, 16), style="fivethirtyeight"):
        self.df = df
        self.figsize = figsize
        self.style = style
        self.colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
        self.poverty_mapping = OrderedDict(
            {1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulerable"}
        )

    def __call__(self, dtype="float"):
        plt.figure(figsize=self.figsize)
        plt.style.use(self.style)
        for i, col in enumerate(self.df.select_dtypes(dtype)):
            ax = plt.subplot(4, 2, i + 1)

            for poverty_level, color in self.colors.items():
                sns.kdeplot(
                    self.df.loc[self.df["Target"] == poverty_level, col].dropna(),
                    ax=ax,
                    color=color,
                    label=self.poverty_mapping[poverty_level],
                )

            plt.title(f"{col.capitalize()} Distribution")
            plt.xlabel(f"{col}")
            plt.ylabel(f"Density")
            plt.legend()
        plt.subplots_adjust(top=2)
