import numpy as np

id_ = ["Id", "idhogar", "Target"]

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']

sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


from collections import Counter
def check_duplicated_features(data):
    """중복된 변수가 있는 지 확인한다.
    """
    x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_
    print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))
    print('We covered every variable: ', len(x) == data.shape[1])


def features_v1(data):
    # sqr 데이터 삭제
    data = remove_sqr_features(data)
    # household 단위 특징 추출
    heads = get_heads_features(data)
    heads = features_hh_v1(heads)
    
    # individual 단위 특징 추출
    ind = data[id_ + ind_bool + ind_ordered]
    ind_agg = feature_ind_v1(ind)
    
    final = merge_left(heads, ind_agg)
    final = add_parent_gender_feature(final, ind)
    
    return final
    
############ household ############
def features_hh_v1(heads):
    
    # correlation이 높은 특징 삭제
    heads = heads.drop(columns=['tamhog', 'hogar_total', 'r4t3'])
    # 가구내 실 거주인 수 - 가족 구성원 수
    heads = add_hhsize_diff(heads)
    # 전기(elec)관련 특징을 ordinal 데이터로 변경
    heads = compress_elec_columns(heads)
    # area2 특징 삭제
    heads = remove_area2_feature(heads)
    # wall의 등급과 관련된 특징들 ordinal로 변경
    heads = onehot_to_ordinal(heads, ord_col="walls", onehot_list=['epared1', 'epared2', 'epared3'])
    # Roof ordinal variable
    heads = onehot_to_ordinal(heads, ord_col="roof", onehot_list=['etecho1', 'etecho2', 'etecho3'])
    # Floor ordinal variable
    heads = onehot_to_ordinal(heads, ord_col="floor", onehot_list=['eviv1', 'eviv2', 'eviv3'])
    heads = add_housequality_feature(heads)
    heads = add_warning_feature(heads)
    heads = add_bonus_feature(heads)
    
    # 인원수 당 장비 수
    heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']
    heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']
    heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']
    heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
    
    return heads
        
    
def remove_sqr_features(data):
    """sqr 데이터를 삭제한다.
    """
    return data.drop(columns = sqr_)


def get_heads_features(data):
    """householde 특징들을 추출하여 반환한다.
    """
    heads = data.loc[data['parentesco1'] == 1, :]
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
    heads = heads.drop(columns = 'area2')
    return heads

def add_housequality_feature(df):
    df["walls+roof+floor"] = df["walls"] + df["roof"] + df["floor"]
    return df


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


def add_bonus_feature(df):
    """냉장고, 컴퓨터, 테블릿, 티비 등의 설비 존재 여부로 새로운 특징(bonus)을 만든다
    """
    
    df['bonus'] = 1 * (df['refrig'] + 
                          df['computer'] + 
                         (df['v18q1'] > 0) + 
                          df['television'])

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
    ind = ind.drop(columns = 'male')
    # onehot 인코딩되어 있는 inst를 ordinal 특징으로 변경
    ind = onehot_to_ordinal(ind, "inst", [c for c in ind if c.startswith('instl')])
    
    # feature construction
    ind['escolari/age'] = ind['escolari'] / ind['age']
    ind['inst/age'] = ind['inst'] / ind['age']
    ind['tech'] = ind['v18q'] + ind['mobilephone']
    
    ind_agg = agg_ind_features(ind)
    ind_agg = drop_high_pcorr_features(ind_agg)
    
    return ind_agg

def agg_ind_features(ind):
    """개별 단위 특징을 가구단위 특징과 통합하기 위해서는 개별 단위 특징을 가구단위로 묶어서 집계할 필요가 있다.     
        가장 간단한 방법으로 `groupby`와 `agg` 매서드를 사용하는 방법이 있다.
    """
    range_ = lambda x: x.max() - x.min()
    range_.__name__ = 'range_'
    
    ind_agg = ind.drop(columns = ['Target', 'Id']).groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])
    
    # Rename the columns
    # 여러 계층으로 나뉜 columns을 단일 계층으로 바꾼다.
    # columns을 단일 리스트로 재설정해주면 된다.
    new_col = []
    for c in ind_agg.columns.levels[0]:
        for stat in ind_agg.columns.levels[1]:
            new_col.append(f'{c}-{stat}')
            
    ind_agg.columns = new_col
    
    return ind_agg
    
########### final ##########
def add_parent_gender_feature(final, ind):
    """
    가장의 성별 데이터 추가
    """
    head_gender = ind.loc[ind['parentesco1'] == 1, ['idhogar', 'female']]
    final = final.merge(head_gender, on = 'idhogar', how = 'left').rename(columns = {'female': 'female-head'})
    return final


# utils
def drop_high_pcorr_features(df):
    """높은 correlation을 갖는 인자들을 제거한다.
    """
    corr_matrix = df.corr()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

    print(f'There are {len(to_drop)} correlated columns to remove.')
    df = df.drop(columns = to_drop)
    ind_feats = list(df.columns)
    
    return df


def merge_left(df_left, df_right):
    return df_left.merge(df_right, on = 'idhogar', how = 'left')


def onehot_to_ordinal(df, ord_col, onehot_list):
    """ binary 형식으로 나누어져 있는 특징들을 ordinal 특징으로 변환한다.
    """
    df[ord_col] = np.argmax(np.array(df[onehot_list]), axis=1)
    df = df.drop(columns=onehot_list)
    return df
