import numpy as np
import pandas as pd

id_ = ["Id", "idhogar", "Target"]

ind_bool = [
    "v18q",
    "dis",
    "male",
    "female",
    "estadocivil1",
    "estadocivil2",
    "estadocivil3",
    "estadocivil4",
    "estadocivil5",
    "estadocivil6",
    "estadocivil7",
    "parentesco1",
    "parentesco2",
    "parentesco3",
    "parentesco4",
    "parentesco5",
    "parentesco6",
    "parentesco7",
    "parentesco8",
    "parentesco9",
    "parentesco10",
    "parentesco11",
    "parentesco12",
    "instlevel1",
    "instlevel2",
    "instlevel3",
    "instlevel4",
    "instlevel5",
    "instlevel6",
    "instlevel7",
    "instlevel8",
    "instlevel9",
    "mobilephone",
    "rez_esc-missing",
]

ind_ordered = ["rez_esc", "escolari", "age"]

hh_bool = [
    "hacdor",
    "hacapo",
    "v14a",
    "refrig",
    "paredblolad",
    "paredzocalo",
    "paredpreb",
    "pisocemento",
    "pareddes",
    "paredmad",
    "paredzinc",
    "paredfibras",
    "paredother",
    "pisomoscer",
    "pisoother",
    "pisonatur",
    "pisonotiene",
    "pisomadera",
    "techozinc",
    "techoentrepiso",
    "techocane",
    "techootro",
    "cielorazo",
    "abastaguadentro",
    "abastaguafuera",
    "abastaguano",
    "public",
    "planpri",
    "noelec",
    "coopele",
    "sanitario1",
    "sanitario2",
    "sanitario3",
    "sanitario5",
    "sanitario6",
    "energcocinar1",
    "energcocinar2",
    "energcocinar3",
    "energcocinar4",
    "elimbasu1",
    "elimbasu2",
    "elimbasu3",
    "elimbasu4",
    "elimbasu5",
    "elimbasu6",
    "epared1",
    "epared2",
    "epared3",
    "etecho1",
    "etecho2",
    "etecho3",
    "eviv1",
    "eviv2",
    "eviv3",
    "tipovivi1",
    "tipovivi2",
    "tipovivi3",
    "tipovivi4",
    "tipovivi5",
    "computer",
    "television",
    "lugar1",
    "lugar2",
    "lugar3",
    "lugar4",
    "lugar5",
    "lugar6",
    "area1",
    "area2",
    "v2a1-missing",
]

hh_ordered = [
    "rooms",
    "r4h1",
    "r4h2",
    "r4h3",
    "r4m1",
    "r4m2",
    "r4m3",
    "r4t1",
    "r4t2",
    "r4t3",
    "v18q1",
    "tamhog",
    "tamviv",
    "hhsize",
    "hogar_nin",
    "hogar_adul",
    "hogar_mayor",
    "hogar_total",
    "bedrooms",
    "qmobilephone",
]

hh_cont = ["v2a1", "dependency", "edjefe", "edjefa", "meaneduc", "overcrowding"]

sqr_ = [
    "SQBescolari",
    "SQBage",
    "SQBhogar_total",
    "SQBedjefe",
    "SQBhogar_nin",
    "SQBovercrowding",
    "SQBdependency",
    "SQBmeaned",
    "agesq",
]

features = [
    "hacdor",
    "hacapo",
    "v14a",
    "refrig",
    "paredblolad",
    "paredzocalo",
    "paredpreb",
    "pisocemento",
    "pareddes",
    "paredmad",
    "paredzinc",
    "paredfibras",
    "paredother",
    "pisomoscer",
    "pisoother",
    "pisonatur",
    "pisonotiene",
    "pisomadera",
    "techozinc",
    "techoentrepiso",
    "techocane",
    "techootro",
    "cielorazo",
    "abastaguadentro",
    "abastaguafuera",
    "abastaguano",
    "sanitario1",
    "sanitario2",
    "sanitario3",
    "sanitario5",
    "sanitario6",
    "energcocinar1",
    "energcocinar2",
    "energcocinar3",
    "energcocinar4",
    "elimbasu1",
    "elimbasu2",
    "elimbasu3",
    "elimbasu4",
    "elimbasu5",
    "elimbasu6",
    "tipovivi1",
    "tipovivi2",
    "tipovivi3",
    "tipovivi4",
    "tipovivi5",
    "computer",
    "television",
    "lugar1",
    "lugar2",
    "lugar3",
    "lugar4",
    "lugar5",
    "lugar6",
    "area1",
    "v2a1-missing",
    "v2a1",
    "dependency",
    "edjefe",
    "edjefa",
    "meaneduc",
    "overcrowding",
    "rooms",
    "r4h1",
    "r4h2",
    "r4h3",
    "r4m1",
    "r4m2",
    "r4m3",
    "r4t1",
    "r4t2",
    "v18q1",
    "tamviv",
    "hhsize",
    "hogar_nin",
    "hogar_adul",
    "hogar_mayor",
    "bedrooms",
    "qmobilephone",
    "hhsize-diff",
    "elec",
    "elec-missing",
    "walls",
    "roof",
    "floor",
    "walls+roof+floor",
    "warning",
    "bonus",
    "phones-per-capita",
    "tablets-per-capita",
    "rooms-per-capita",
    "rent-per-capita",
    "v18q-min",
    "v18q-sum",
    "v18q-count",
    "v18q-std",
    "v18q-range_",
    "dis-min",
    "dis-max",
    "dis-sum",
    "female-min",
    "female-max",
    "female-sum",
    "female-std",
    "female-range_",
    "estadocivil1-min",
    "estadocivil1-max",
    "estadocivil1-sum",
    "estadocivil2-min",
    "estadocivil2-max",
    "estadocivil2-std",
    "estadocivil3-min",
    "estadocivil3-max",
    "estadocivil3-std",
    "estadocivil4-min",
    "estadocivil4-max",
    "estadocivil5-min",
    "estadocivil5-max",
    "estadocivil5-sum",
    "estadocivil6-min",
    "estadocivil6-max",
    "estadocivil7-min",
    "estadocivil7-max",
    "estadocivil7-sum",
    "estadocivil7-std",
    "parentesco1-min",
    "parentesco1-max",
    "parentesco1-std",
    "parentesco2-min",
    "parentesco2-max",
    "parentesco2-std",
    "parentesco3-min",
    "parentesco3-max",
    "parentesco3-sum",
    "parentesco4-min",
    "parentesco4-max",
    "parentesco4-sum",
    "parentesco5-min",
    "parentesco5-max",
    "parentesco6-min",
    "parentesco6-max",
    "parentesco6-sum",
    "parentesco7-min",
    "parentesco7-max",
    "parentesco8-min",
    "parentesco8-max",
    "parentesco9-min",
    "parentesco9-max",
    "parentesco9-sum",
    "parentesco10-min",
    "parentesco10-max",
    "parentesco11-min",
    "parentesco11-max",
    "parentesco11-sum",
    "parentesco12-min",
    "parentesco12-max",
    "parentesco12-sum",
    "mobilephone-min",
    "mobilephone-std",
    "mobilephone-range_",
    "rez_esc-min",
    "rez_esc-max",
    "escolari-min",
    "escolari-max",
    "escolari-sum",
    "escolari-std",
    "escolari-range_",
    "age-min",
    "age-max",
    "age-sum",
    "age-std",
    "age-range_",
    "inst-max",
    "inst-std",
    "inst-range_",
    "escolari/age-min",
    "escolari/age-max",
    "escolari/age-sum",
    "escolari/age-std",
    "escolari/age-range_",
    "inst/age-max",
    "inst/age-std",
    "inst/age-range_",
    "tech-min",
    "tech-sum",
    "tech-std",
    "tech-range_",
    "female-head",
]

from collections import Counter


def check_duplicated_features(data):
    """중복된 변수가 있는 지 확인한다."""
    x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_
    print("There are no repeats: ", np.all(np.array(list(Counter(x).values())) == 1))
    print("We covered every variable: ", len(x) == data.shape[1])


def preprocessing_train_test_and_merge(train, test):

    # 가족구성원들의 poverty level을 세대주와 통일
    train = correct_poverty_levels(train)

    mapping = {"yes": 1, "no": 0}
    columns = ["dependency", "edjefa", "edjefe"]
    replace_object_value = ReplaceValues(mapping)

    train = replace_object_value(train, columns)
    test = replace_object_value(test, columns)

    test["Target"] = np.nan
    data = pd.concat([train, test], axis=0, ignore_index=True)

    return data


def features_v1(data):

    # sqr 데이터 삭제
    data = remove_sqr_features(data)

    # null 값 채우기
    data = fill_nan_with_zero(data, "v18q1")
    data = fill_null_v2a1(data)
    data = fill_null_rez_esc(data)

    # household 단위 특징 추출
    heads = get_heads_features(data)
    heads = features_hh_v1(heads)

    # individual 단위 특징 추출
    ind = data[id_ + ind_bool + ind_ordered]
    ind_agg = feature_ind_v1(ind)

    final = merge_left(heads, ind_agg)
    final = add_parent_gender_feature(final, ind)

    # train_set, test_set, train_labels = split_train_test(final)

    #return train_set, test_set, train_labels
    return final


########### fill null data ##########


def fill_nan_with_zero(df, col):
    df[col] = df[col].fillna(0)
    return df


def fill_null_v2a1(df):
    df.loc[(df["tipovivi1"] == 1), "v2a1"] = 0
    df["v2a1-missing"] = df["v2a1"].isnull()
    return df


def fill_null_rez_esc(data):
    """
    https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403
    위 링크에 따르면 `rez_esc` 특징은 7~19세 사이의 나이대에 해당하는 샘플에서만 정의된 것을 알 수 있다.
    따라서 해당 나이 범위에 해당하지 않은 샘플의 결측치는 0으로 설정할 수 있다.
    그에 해당하지 않는 샘플의 경우 이전과 같이 결측치임을 알려주는 새로운 열을 추가한 후 다른 숫자로 대체한다.
    """
    data.loc[((data["age"] > 19) | (data["age"] < 7)) & (data["rez_esc"].isnull()), "rez_esc"] = 0
    data["rez_esc-missing"] = data["rez_esc"].isnull()

    #   competition discussion에 따르면 `rez_esc` 특징의 최대값은 5이다.
    #   따라서 해당 값보다 큰 값을 갖는 샘플은 5로 값을 제한한다.
    data.loc[data["rez_esc"] > 5, "rez_esc"] = 5

    return data


############ household ############
def features_hh_v1(heads):

    # correlation이 높은 특징 삭제
    heads = heads.drop(columns=["tamhog", "hogar_total", "r4t3"])
    # 가구내 실 거주인 수 - 가족 구성원 수
    heads = add_hhsize_diff(heads)
    # 전기(elec)관련 특징을 ordinal 데이터로 변경
    heads = compress_elec_columns(heads)
    # area2 특징 삭제
    heads = remove_area2_feature(heads)
    # wall의 등급과 관련된 특징들 ordinal로 변경
    heads = onehot_to_ordinal(heads, ord_col="walls", onehot_list=["epared1", "epared2", "epared3"])
    # Roof ordinal variable
    heads = onehot_to_ordinal(heads, ord_col="roof", onehot_list=["etecho1", "etecho2", "etecho3"])
    # Floor ordinal variable
    heads = onehot_to_ordinal(heads, ord_col="floor", onehot_list=["eviv1", "eviv2", "eviv3"])
    heads = add_housequality_feature(heads)
    heads = add_warning_feature(heads)
    heads = add_bonus_feature(heads)

    # 인원수 당 장비 수
    heads["phones-per-capita"] = heads["qmobilephone"] / heads["tamviv"]
    heads["tablets-per-capita"] = heads["v18q1"] / heads["tamviv"]
    heads["rooms-per-capita"] = heads["rooms"] / heads["tamviv"]
    heads["rent-per-capita"] = heads["v2a1"] / heads["tamviv"]

    return heads


def remove_sqr_features(data):
    """sqr 데이터를 삭제한다."""
    return data.drop(columns=sqr_)


def get_heads_features(data):
    """householde 특징들을 추출하여 반환한다."""
    heads = data.loc[data["parentesco1"] == 1, :]
    heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

    return heads


def compress_elec_columns(heads):
    """전기와 관련된 특징을 ordinal 특징으로 변경한다.
    전기 수급 방식에 따라 가난의 정도를 어느정도 반영할 수 있다.
    """
    elec = []

    for _, row in heads.iterrows():
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

    heads["elec"] = elec
    heads["elec-missing"] = heads["elec"].isnull()
    heads = heads.drop(columns=["noelec", "coopele", "public", "planpri"])
    return heads


def remove_area2_feature(heads):
    """`area2` 특징은 집이 시골지역에 있는지를 나타낸다.
    위 특징은 집이 도시에 있는지를 나타내는 특징(`area1`)과 같은 정보를 가지므로 제거한다.
    """
    heads = heads.drop(columns="area2")
    return heads


def add_housequality_feature(df):
    df["walls+roof+floor"] = df["walls"] + df["roof"] + df["floor"]
    return df


def add_warning_feature(df):
    """toilet, electricity, floor, water service, ceiling 등 설비가 없는 가구에 대해 -1 값을 갖는 특징들을 합해 warning 특징을 만든다.

    Args:
        df (_type_): _description_
    """
    df["warning"] = 1 * (
        df["sanitario1"]
        + (df["elec"] == 0)
        + df["pisonotiene"]
        + df["abastaguano"]
        + (df["cielorazo"] == 0)
    )

    return df


def add_bonus_feature(df):
    """냉장고, 컴퓨터, 테블릿, 티비 등의 설비 존재 여부로 새로운 특징(bonus)을 만든다"""

    df["bonus"] = 1 * (df["refrig"] + df["computer"] + (df["v18q1"] > 0) + df["television"])

    return df


def add_hhsize_diff(heads, is_viz=True):
    """
    가족 구성원 수보다 더 많은 수의 인원이 한 가구에 사는 샘플이 존재하는 것을 알 수 있다.
    위 정보를 통해 두 값의 차이로 새로운 특징을 만들어 낼 수 있다.
    """
    heads["hhsize-diff"] = heads["tamviv"] - heads["hhsize"]

    return heads


########## individual ##########


def feature_ind_v1(ind):

    # female과 같은 정보를 담고 있는 male 특징 삭제
    ind = ind.drop(columns="male")
    # onehot 인코딩되어 있는 inst를 ordinal 특징으로 변경
    ind = onehot_to_ordinal(ind, "inst", [c for c in ind if c.startswith("instl")])

    # feature construction
    ind["escolari/age"] = ind["escolari"] / ind["age"]
    ind["inst/age"] = ind["inst"] / ind["age"]
    ind["tech"] = ind["v18q"] + ind["mobilephone"]

    ind_agg = agg_ind_features(ind)
    ind_agg = drop_high_pcorr_features(ind_agg)

    return ind_agg


def agg_ind_features(ind):
    """개별 단위 특징을 가구단위 특징과 통합하기 위해서는 개별 단위 특징을 가구단위로 묶어서 집계할 필요가 있다.
    가장 간단한 방법으로 `groupby`와 `agg` 매서드를 사용하는 방법이 있다.
    """
    range_ = lambda x: x.max() - x.min()
    range_.__name__ = "range_"

    ind_agg = (
        ind.drop(columns=["Target", "Id", "rez_esc-missing"])
        .groupby("idhogar")
        .agg(["min", "max", "sum", "count", "std", range_])
    )

    # Rename the columns
    # 여러 계층으로 나뉜 columns을 단일 계층으로 바꾼다.
    # columns을 단일 리스트로 재설정해주면 된다.
    new_col = []
    for c in ind_agg.columns.levels[0]:
        for stat in ind_agg.columns.levels[1]:
            new_col.append(f"{c}-{stat}")

    ind_agg.columns = new_col

    return ind_agg


########### final ##########
def add_parent_gender_feature(final, ind):
    """
    가장의 성별 데이터 추가
    """
    head_gender = ind.loc[ind["parentesco1"] == 1, ["idhogar", "female"]]
    final = final.merge(head_gender, on="idhogar", how="left").rename(
        columns={"female": "female-head"}
    )
    return final


# utils
def drop_high_pcorr_features(df):
    """높은 correlation을 갖는 인자들을 제거한다."""
    corr_matrix = df.corr()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

    print(f"There are {len(to_drop)} correlated columns to remove.")
    df = df.drop(columns=to_drop)
    ind_feats = list(df.columns)

    return df


def merge_left(df_left, df_right):
    return df_left.merge(df_right, on="idhogar", how="left")


def onehot_to_ordinal(df, ord_col, onehot_list):
    """binary 형식으로 나누어져 있는 특징들을 ordinal 특징으로 변환한다."""
    df[ord_col] = np.argmax(np.array(df[onehot_list]), axis=1)
    df = df.drop(columns=onehot_list)
    return df


class ReplaceValues:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, df, columns):
        # Fill in the values with the correct mapping
        for c in columns:
            df[f"{c}"] = df[f"{c}"].replace(self.mapping).astype(np.float64)

        return df


def num_different_poverty_level(df):
    """
    가족 구성원들 중 세대주와 poverty level이 다른 구성원이 존재하는 household id를 반환한다.
    """
    all_equal = df.groupby("idhogar")["Target"].apply(lambda x: x.nunique() == 1)
    not_equal = all_equal[all_equal != True]
    print(
        f"There are {len(not_equal)} households where the family members do not all have the same target"
    )

    return not_equal


def correct_poverty_levels(df):
    """
    세대주와 poreverty level이 다른 가족 구성원들의 poverty level 를 일치시킨다.
    """
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

    return df


def split_train_test(final):
    """
    처리한 데이터를 다시 train과 test set으로 나눈다.
    """
    # Labels for training
    train_labels = np.array(list(final[final["Target"].notnull()]["Target"].astype(np.uint8)))

    # Extract the training data
    train_set = final[final["Target"].notnull()].drop(columns=["Id", "idhogar", "Target"])
    test_set = final[final["Target"].isnull()].drop(columns=["Id", "idhogar", "Target"])

    return train_set, test_set, train_labels
