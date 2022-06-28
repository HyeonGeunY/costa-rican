import pandas as pd
import numpy as np
from costa_rican.utils.visualization import PlotCategoricals


class ReplaceValues:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, df, columns):
        # Fill in the values with the correct mapping
        for c in columns:
            df[f"{c}"] = df[f"{c}"].replace(self.mapping).astype(np.float64)

        return df


def get_num_without_household(df):
    households_leader = df.groupby("idhogar")["parentesco1"].sum()
    households_no_head = df.loc[
        df["idhogar"].isin(households_leader[households_leader == 0].index), :
    ]
    print(f"There are {households_no_head['idhogar'].nunique()} households without a head.")

    return households_no_head


def get_missing_val_info(df):
    missing = pd.DataFrame(df.isnull().sum()).rename(columns={0: "total"})
    missing["percent"] = missing["total"] / len(df)
    return missing


def get_null_by_group(df, group, col):
    print(df.groupby(group)[col].apply(lambda x: x.isnull().sum()))



def get_features_over_corr(df, corr_num):
    """각 특징들에 대해서 상관관계가 95%가 넘는 특징이 있는 특징들을 담은 리스트를 반환한다."""
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > corr_num)]
    print(to_drop)

    return corr_matrix


def get_count_of_housequality(df):
    counts = (
        pd.DataFrame(df.groupby(["walls+roof+floor"])["Target"].value_counts(normalize=True))
        .rename(columns={"Target": "Normalized Count"})
        .reset_index()
    )
    return counts


def get_pcorrs(df):
    """train data에 대해서 pearson corrletion을 구해서 반환한다.
    """
    train_heads = df.loc[df['Target'].notnull(), :].copy()
    
    pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target': 'pcorr'}).reset_index()
    pcorrs = pcorrs.rename(columns = {'index': 'feature'})
    
    print('Most negatively correlated variables:')
    print(pcorrs.head())

    print('\nMost positively correlated variables:')
    print(pcorrs.dropna().tail())
    
    return pcorrs

from scipy.stats import spearmanr

def get_scorrs(df):
    """
    각 특징들에 대해 spearman correlation 값을 구한다.
    """
    
    feats = []
    scorr = []
    pvalues = []
    
    train_heads = df.loc[df['Target'].notnull(), :].copy()
    # Iterate through each column
    for c in df:
        # Only valid for numbers
        if df[c].dtype != 'object':
            feats.append(c)
            
            # Calculate spearman correlation
            scorr.append(spearmanr(train_heads[c], train_heads['Target']).correlation)
            pvalues.append(spearmanr(train_heads[c], train_heads['Target']).pvalue)

    scorrs = pd.DataFrame({'feature': feats, 'scorr': scorr, 'pvalue': pvalues}).sort_values('scorr')
    
    return scorrs