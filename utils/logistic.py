import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class LogisticReg:

    def __init__(self, f_number=2, c=10, k=0.1):
        self.n = f_number
        self.w = np.array([[0]])
        self.C = c
        self.k = k
        for i in range(self.n - 1):
            self.w = np.vstack((self.w, [0]))
        self.def_w = self.w
        pass

    def fit(self, X_train, y_train, regulariz=False):
        J = 0
        m = y_train.size
        self.w = self.def_w
        for i in range(10000):
            margin = (X_train.dot(self.w).transpose().multiply(-y_train))
            J = np.log(1 + np.exp(margin)).sum(axis=1) / m + np.linalg.norm(self.w)**2 * self.C * 0.5
            last_w = self.w
            brack = (1 - 1/(np.exp(margin) + 1)).transpose()
            xy = X_train.transpose().multiply(y_train)
            deriv = xy.dot(brack)
            reg_term = self.k*self.C*self.w
            self.w = self.w + deriv*self.k/m  # - self.k*self.C*self.w
            no_reg_str = 'out'
            if regulariz:
                self.w = self.w - reg_term
            if np.linalg.norm(last_w - self.w) <= 0.00001:
                print(f'\nWith{no_reg_str if not regulariz else str()} regularization ideal w is: \n{self.w} \nwas found on {i} iteration!')
                break
        pass

    def getassesment(self, X_train, y_train):
        y_pred = 1/(1 + np.exp(-X_train.dot(self.w)))
        assesment = roc_auc_score(y_train, y_pred)
        print(f'ROC-AUC score is: {assesment}')
        return assesment

    @classmethod
    def loadData(cls, path, feature_number=2):
        data = pd.read_csv(path, header=None)
        y = data[0]
        X = data.drop(0, axis=1)
        return [X, y]
