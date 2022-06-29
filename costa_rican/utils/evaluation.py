from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score


def make_custom_scorer():
    """
    custom scorer를 반환한다.
    """
    return make_scorer(f1_score, greater_is_better=True, average = 'macro')


class CVScore():
    def __init__(self, clf, seed=2022, model_params=None, scorer_params=None, cv=10):
        model_params['random_state'] = seed
        self.clf = clf(**model_params)
        self.scorer = make_scorer(**scorer_params)
        self.cv = cv
    
    def train(self, x_train, y_train):
        cv_score = cross_val_score(self.clf, x_train, y_train, cv=self.cv, scoring=self.scorer)
        print(f"{str(self.clf.__class__).rsplit('.', 1)[-1][:-2]}")
        print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')
        
        
### scorer ###
scorer_params = {
    'score_func': f1_score,
    'greater_is_better': True,
    'average': 'macro'
}

### RF ###

rf_params = {
    'n_estimators': 100,
    'random_state': 10,
    'n_jobs': -1}