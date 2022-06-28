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


def num_different_poverty_level(df):
    all_equal = df.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
    not_equal = all_equal[all_equal != True]
    print(
        f"There are {len(not_equal)} households where the family members do not all have the same target"
    )

    return not_equal


def get_num_without_household(df):
    households_leader = df.groupby("idhogar")["parentesco1"].sum()
    households_no_head = df.loc[
        df["idhogar"].isin(households_leader[households_leader == 0].index), :
    ]
    print(f"There are {households_no_head['idhogar'].nunique()} households without a head.")

    return households_no_head


def correct_poverty_levels(df):
    print("Before")
    not_equal = num_different_poverty_level(df)
    for household in not_equal.index:
        true_target = int(df[(df["idhogar"] == household) & (df["parentesco1"] == 1.0)]["Target"])

        df.loc[df["idhogar"] == household, "Target"] = true_target

    all_equal = df.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
    not_equal = all_equal[all_equal != True]
    print("After")
    print(
        f"There are {len(not_equal)} households where the family members do not all have the same target"
    )


def get_missing_val_info(df):
    missing = pd.DataFrame(df.isnull().sum()).rename(columns={0: "total"})
    missing["percent"] = missing["total"] / len(df)
    return missing


def get_null_by_group(df, group, col):
    print(df.groupby(group)[col].apply(lambda x: x.isnull().sum()))


def fill_nan_with_zero(df, col):
    df[col] = df[col].fillna(0)
    return df


def fill_null_v2a1(df):
    df.loc[(df["tipovivi1"] == 1), "v2a1"] = 0
    df["v2a1-missing"] = df["v2a1"].isnull()
    return df


def get_features_over_corr(df, corr_num):
    """각 특징들에 대해서 상관관계가 95%가 넘는 특징이 있는 특징들을 담은 리스트를 반환한다."""
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > corr_num)]
    print(to_drop)

    return corr_matrix


def get_hhsize_diff(df, is_viz=True):
    df["hhsize-diff"] = df["tamviv"] - df["hhsize"]

    if is_viz:
        plot_categoricals = PlotCategoricals()
        plot_categoricals("hhsize-diff", "Target", df)

    return df


def compress_elec_columns(df):
    elec = []

    for i, row in df.iterrows():
        if row["noelec"] == 1:
            elec.append(0)
        elif row["coopele"] == 1:
            elec.append(1)
        elif row["public"] == 1:
            elec.append(2)
        elif row["planpri"] == 1:
            elec.append(3)
        else:
            elec.append(np.nan)

    df["elec"] = elec
    df["elec-missing"] = df["elec"].isnull()
    df = df.drop(columns=["noelec", "coopele", "public", "planpri"])
    return df


def onehot_to_ordinal(df, ord_col, onehot_list):
    df[ord_col] = np.argmax(np.array(df[onehot_list]), axis=1)
    df = df.drop(columns=onehot_list)
    return df


def add_housequality_feature(df):
    df["walls+roof+floor"] = df["walls"] + df["roof"] + df["floor"]
    return df


def get_count_of_housequality(df):
    counts = (
        pd.DataFrame(df.groupby(["walls+roof+floor"])["Target"].value_counts(normalize=True))
        .rename(columns={"Target": "Normalized Count"})
        .reset_index()
    )
    return counts


def add_warning_feature(df):
    """toilet, electricity, floor, water service, ceiling 등 설비가 없는 가구에 대해 -1 값을 갖는 특징들을 합해 warning 특징을 만든다.

    Args:
        df (_type_): _description_
    """
    df['warning'] = 1 * (df['sanitario1'] + 
                        (df['elec'] == 0) + 
                         df['pisonotiene'] + 
                         df['abastaguano'] + 
                        (df['cielorazo'] == 0))
    
    return df


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