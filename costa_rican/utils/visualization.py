from termios import VMIN
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


class KdePlotByTargetAndColumn:
    def __init__(self, df, figsize=(20, 16), style="fivethirtyeight"):
        self.df = df
        self.figsize = figsize
        self.style = style
        self.colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
        self.poverty_mapping = OrderedDict(
            {1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulerable"}
        )

    def __call__(self, columns=[]):
        plt.figure(figsize=self.figsize)
        plt.style.use(self.style)
        for i, col in enumerate(columns):
            ax = plt.subplot(len(columns), 1, i + 1)

            for poverty_level, color in self.colors.items():
                sns.kdeplot(
                    self.df.loc[self.df["Target"] == poverty_level, col].dropna(),
                    ax=ax,
                    color=color,
                    label=self.poverty_mapping[poverty_level],
                )

            plt.title(f"{col.capitalize()} Distribution")
            plt.xlabel(f"{col}")
            plt.ylabel("Density")
            plt.legend()
        plt.subplots_adjust(top=2)


class ShowLabelDistribution:
    def __init__(self, edgecolor="k", linewidth=2, figsize=(8, 6)):

        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.figsize = figsize
        self.colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
        self.poverty_mapping = OrderedDict(
            {1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulerable"}
        )

    def __call__(self, df, target="Target"):
        # train 데이터에서 세대주의 target과 household 열 추출
        train_labels = df.loc[
            (df[target].notnull()) & (df["parentesco1"] == 1), [target, "idhogar"]
        ]
        label_counts = train_labels[target].value_counts().sort_index()
        label_counts.plot.bar(
            figsize=self.figsize,
            color=self.colors.values(),
            edgecolor=self.edgecolor,
            linewidth=self.linewidth,
        )

        # Formatting
        plt.xlabel("Poverty Level")
        plt.ylabel("Count")
        plt.xticks(
            [x - 1 for x in self.poverty_mapping.keys()],
            list(self.poverty_mapping.values()),
            rotation=60,
        )
        plt.title("Poverty Level Breakdown")

        return label_counts


class BarValueCounts:
    def __init__(self, color="blue", edgecolor="k", linewidth=2, figsize=(8, 6)):
        self.figsize = figsize
        self.color = color
        self.edgecolor = edgecolor
        self.linewidth = linewidth

    def __call__(self, df, col, heads_only=True):
        """_summary_
        특정 열의 클래스별 숫자를 plot 한다.
        Args:
            col (_type_): _description_
            heads_only (bool, optional): 세대주 샘플만 보기(가구 단위 특징의 경우 사용). Defaults to False.
        """

        if heads_only:
            df = df.loc[df["parentesco1"] == 1].copy()

        plt.figure(figsize=self.figsize)
        df[col].value_counts().sort_index().plot.bar(
            color=self.color, edgecolor=self.edgecolor, linewidth=self.linewidth
        )

        plt.xlabel(f"{col}")
        plt.title(f"{col} value counts")
        plt.ylabel("Count")
        plt.show()


class NullCountsByCols:
    def __init__(
        self, color="green", edgecolor="k", linewidth=2, rotation=60, title_size=18, figsize=(10, 8)
    ):
        self.color = color
        self.edgecolor = edgecolor
        self.figsize = figsize
        self.linewidth = linewidth
        self.rotation = rotation
        self.title_size = title_size

    def __call__(self, df, null_col, by_col_key, xtickslabels, title):
        own_variable = [x for x in df if x.startswith(by_col_key)]
        df.loc[df[null_col].isnull(), own_variable].sum().plot.bar(
            figsize=self.figsize, color=self.color, ecolor=self.edgecolor, linewidth=self.linewidth
        )

        plt.xticks(range(len(own_variable)), xtickslabels, rotation=self.rotation)
        plt.title(title, size=self.title_size)


class PlotCategoricals:
    def __init__(
        self,
        annotation_color="navy",
        color="lightgreen",
        marker="o",
        edgecolor="k",
        alpha=0.6,
        linewidth=1.5,
        figsize=(14, 10),
    ):

        self.figsize = figsize
        self.color = color
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.marker = marker
        self.linewidth = linewidth
        self.annotation_color = annotation_color

    def __call__(self, x, y, df, annotate=True):
        raw_counts = pd.DataFrame(df.groupby(y)[x].value_counts(normalize=False))
        raw_counts = raw_counts.rename(columns={x: "raw_count"})

        counts = pd.DataFrame(df.groupby(y)[x].value_counts(normalize=True))
        counts = counts.rename(columns={x: "normalized_count"}).reset_index()
        counts["percent"] = 100 * counts["normalized_count"]

        # Add the raw count
        counts["raw_count"] = list(raw_counts["raw_count"])

        plt.figure(figsize=self.figsize)
        plt.scatter(
            counts[x],
            counts[y],
            edgecolor=self.edgecolor,
            color=self.color,
            s=100 * np.sqrt(counts["raw_count"]),
            marker=self.marker,
            alpha=self.alpha,
            linewidth=self.linewidth,
        )

        if annotate:
            for i, row in counts.iterrows():
                plt.annotate(
                    f"{round(row['percent'], 1)}%",
                    xy=(row[x] - (1 / counts[x].nunique()), row[y] - (0.15 / counts[y].nunique())),
                    color=self.annotation_color,
                )

        plt.yticks(counts[y].unique())
        plt.xticks(counts[x].unique())

        sqr_min = int(np.sqrt(raw_counts["raw_count"].min()))
        sqr_max = int(np.sqrt(raw_counts["raw_count"].max()))

        msizes = list(range(sqr_min, sqr_max, int((sqr_max - sqr_min) / 5)))
        markers = []

        for size in msizes:
            markers.append(
                plt.scatter(
                    [],
                    [],
                    s=100 * size,
                    label=f"{int(round(np.square(size) / 100) * 100)}",
                    color=self.color,
                    alpha=self.alpha,
                    edgecolor=self.edgecolor,
                    linewidth=self.linewidth,
                )
            )

        plt.legend(
            handles=markers,
            title="Counts",
            labelspacing=3,
            handletextpad=2,
            fontsize=16,
            loc=(1.10, 0.19),
        )
        plt.annotate(
            f"* Size represents raw count while % is for a given y value.",
            xy=(0, 1),
            xycoords="figure points",
            size=10,
        )
        plt.xlim(
            (
                counts[x].min() - (6 / counts[x].nunique()),
                counts[x].max() + (6 / counts[x].nunique()),
            )
        )
        plt.ylim(
            (
                counts[y].min() - (4 / counts[y].nunique()),
                counts[y].max() + (4 / counts[y].nunique()),
            )
        )
        plt.grid(None)
        plt.xlabel(f"{x}")
        plt.ylabel(f"{y}")
        plt.title(f"{y} vs {x}")


class HeatmapSeabornWithThreshold:
    def __init__(self, annot=True, fmt=".3f", cmap=plt.cm.autumn_r):
        self.annot = annot
        self.fmt = fmt
        self.cmap = cmap
        

    def __call__(self, feature, corr_matrix, thres=0.9):
        sns.heatmap(
            corr_matrix.loc[corr_matrix[feature].abs() > thres, corr_matrix[feature].abs() > thres],
            annot=self.annot,
            cmap=plt.cm.autumn_r,
            fmt=self.fmt,
        )



class HeatmapSeaborn:
    
    def __init__(self, annot=True, fmt=".3f", cmap=plt.cm.autumn_r, fontsize=18, figsize=(12, 12), vmin=-0.5, vmax=0.8, center=0):
        self.annot = annot
        self.cmap = cmap
        self.fontsize = fontsize
        self.figsize = figsize
        self.vmin = vmin
        self.vmax = vmax
        self.center = center

    def __call__(self, corr_matrix):
        plt.rcParams['font.size'] = self.fontsize
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, vmin=self.vmin, vmax=self.vmax, center=self.center, 
            cmap = plt.cm.RdYlGn_r, annot=self.annot);



class LmPlot:
    def __init__(self, fig_reg=False, height=8, x_jitter=0.05, y_jitter=0.05):
        self.fig_reg = fig_reg
        self.height = height
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        

    def __call__(self, x, y, df):
        
        sns.lmplot(x=x, y=y, data=df, fit_reg=self.fig_reg, height=self.height, x_jitter=self.x_jitter, y_jitter=self.y_jitter)
        
        plt.title(f"{y} versus {x}")


class ViolinByTarget():
    def __init__(self, figsize=(10, 6)):
        self.figsize=figsize
    
    def __call__(self, x, y, data, title=None):
        plt.figure(figsize=self.figsize)
        sns.violinplot(x=x, y=y, data=data)
        if title:
            plt.title(title)
        else:
            plt.title(f"{y} vs {x} variables")
            
            
from scipy.stats import spearmanr

class PlotCorrs():
    def __init__(self, figsize=(8, 6), fit_reg=False):
        self.figsize=figsize
        self.fit_reg=fit_reg
    
    def __call__(self, x, y):
        plt.figure(figsize=self.figsize)
        spr = spearmanr(x, y).correlation
        pcr = np.corrcoef(x, y)[0, 1]
        df = pd.DataFrame({'x': x, 'y': y})
        sns.regplot(x='x', y='y', data=df, fit_reg=False);
        plt.title(f'Spearman: {round(spr, 2)} Pearson: {round(pcr, 2)}')
        
        
class BoxByTarget():
    def __init__(self, figsize=(10, 6), target_xticks=False):
        self.figsize = figsize
        self.target_xticks = target_xticks
        self.colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
        self.poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})
        
    def __call__(self, x, y, data, title=None, hue=None):
        plt.figure(figsize=self.figsize)
        sns.boxplot(x=x, y=y, data=data, hue=hue)
        if title:
            plt.title(title)
        else:
            plt.title(f"{y} vs {x} variables")
            
        if self.target_xticks:
            plt.xticks(range(len(self.poverty_mapping)), self.poverty_mapping.values())