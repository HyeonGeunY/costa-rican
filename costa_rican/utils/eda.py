import pandas as pd
import numpy as np


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
