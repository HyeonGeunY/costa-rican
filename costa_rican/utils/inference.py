import pandas as pd
import numpy as np 

def submit(model, train, train_labels, test, test_ids, submission_base):
    """Train and test a model on the dataset
    """
    model.fit(train, train_labels)
    predictions = model.predict(test)
    predictions = pd.DataFrame({'idhogar': test_ids,
                                'Target': predictions})
    
    submission = submission_base.merge(predictions, 
                                       on = 'idhogar',
                                       how = 'left').drop(columns = ['idhogar'])
    
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    
    return submission