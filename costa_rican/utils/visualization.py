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


class ShowLabelDistribution():
    def __init__(self, edgecolor='k', linewidth=2, figsize=(8, 6)):
        
        self.edgecolor=edgecolor
        self.linewidth=linewidth
        self.figsize=figsize
        self.colors = OrderedDict({1: "red", 2: "orange", 3: "blue", 4: "green"})
        self.poverty_mapping = OrderedDict(
            {1: "extreme", 2: "moderate", 3: "vulnerable", 4: "non vulerable"}
        )
    
    def __call__(self, df, target="Target"):
        # train 데이터에서 세대주의 target과 household 열 추출
        train_labels = df.loc[(df[target].notnull()) & (df['parentesco1'] == 1), [target, 'idhogar']]
        label_counts = train_labels[target].value_counts().sort_index()
        label_counts.plot.bar(figsize=self.figsize, color=self.colors.values(), edgecolor=self.edgecolor, linewidth=self.linewidth)

        # Formatting
        plt.xlabel('Poverty Level')
        plt.ylabel('Count')
        plt.xticks([x - 1 for x in self.poverty_mapping.keys()],
                   list(self.poverty_mapping.values()), rotation=60)
        plt.title('Poverty Level Breakdown')
        
        return label_counts
    
    
class BarValueCounts():
    
    def __init__(self, color='blue', edgecolor='k', linewidth=2, figsize=(8, 6)):
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
            df = df.loc[df['parentesco1'] == 1].copy()
            
        plt.figure(figsize = self.figsize)
        df[col].value_counts().sort_index().plot.bar(color=self.color, edgecolor=self.edgecolor, linewidth=self.linewidth)
        
        plt.xlabel(f"{col}")
        plt.title(f"{col} value counts")
        plt.ylabel('Count')
        plt.show()
        

class NullCountsByCols():
    def __init__(self, color='green', edgecolor='k', linewidth=2, rotation=60, title_size=18, figsize=(10, 8)):
        self.color = color
        self.edgecolor = edgecolor
        self.figsize = figsize
        self.linewidth = linewidth
        self.rotation = rotation
        self.title_size = title_size
    
    def __call__(self, df, null_col, by_col_key, xtickslabels, title):
        own_variable = [x for x in df if x.startswith(by_col_key)]
        df.loc[df[null_col].isnull(), own_variable].sum().plot.bar(
            figsize=self.figsize,
            color=self.color,
            ecolor=self.edgecolor,
            linewidth=self.linewidth
        )
        
        plt.xticks(range(len(own_variable)), xtickslabels, rotation=self.rotation)
        plt.title(title, size=self.title_size)
        