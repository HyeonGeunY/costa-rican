from costa_rican.utils.eda import onehot_to_ordinal
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


def feature_ind_v1(ind):
    
    # female과 같은 정보를 담고 있는 male 특징 삭제
    ind = ind.drop(columns = 'male')
    # onehot 인코딩되어 있는 inst를 ordinal 특징으로 변경
    ind = onehot_to_ordinal(ind, "inst", [c for c in ind if c.startswith('instl')])
    
    # feature construction
    ind['escolari/age'] = ind['escolari'] / ind['age']
    ind['inst/age'] = ind['inst'] / ind['age']
    ind['tech'] = ind['v18q'] + ind['mobilephone']
    

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