from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from costa_rican.utils.feature_engineering import features

import pandas as pd

class SklearnWrapper(object):
    def __init__(self, clf, seed=2022, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
    
    def get_feature_importances(self):
        feature_importances = pd.DataFrame({'feature': features, 'importance': self.clf.feature_importances_})
        return feature_importances
        
# utils
def fill_null_and_scaling(train_set, test_set):
    """
    Pipeline 함수를 이용하여 scaling과 결측값 처리를 같이 이어서 해준다.
    """

    pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')), 
                        ('scaler', MinMaxScaler())])

    # Fit and transform training data
    train_set = pipeline.fit_transform(train_set)
    test_set = pipeline.transform(test_set)
    
    return train_set, test_set


### params ###

rf_params = {
    'n_estimators': 100,
    'random_state': 10,
    'n_jobs': -1
    }


