import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from include.Classifiers import getClassifier

class Model:
    def __init__(self, genes, params, data):
        self.genes = genes
        self.params = params
        self.data = data
        self.classifier = []

    def train(self):
        df_out = pd.DataFrame()
        df_out['Actual'] = self.data.y_val
        df_out.reset_index(drop=True,inplace=True)
        
        #print(f'Training: {self.params["classifier_name"]}')
        #try:
        #    print(self.genes['m_base_estimator'])
        #except:
        #    pass
        self.classifier = getClassifier(self.params['classifier_name'], self.genes, self.genes['b_use_model_without_parameter'])
        self.classifier.fit(X=self.data.X_train, y=self.data.y_train)

        acc = self.evaluate(self.data.X_val, self.data.y_val)
        return acc

    def evaluate(self, X, y):
        pred = self.classifier.predict(X)
        acc = np.float(f'{accuracy_score(y, pred) * 100:0.1f}')

        return acc