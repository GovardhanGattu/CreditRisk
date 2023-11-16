import os
import sys
from src.exception import CustomeException
from src.logger import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from src.utils import saveobject,evaluatemodel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig():
    model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.modeltrainerconfig=ModelTrainerConfig()

    def initiate_model_training(self,train_data,test_data):
        try:
            X_train,Y_train,X_test,Y_test =(
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
                )
            
            param_grid = {'penalty':['l1','l2'], 'solver':['liblinear'],'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
            grid_lr_clf = GridSearchCV(LogisticRegression(), param_grid, scoring = 'accuracy',  verbose = 3, cv = 3)
            grid_lr_clf.fit(X_train, Y_train)
            optimized_clf = grid_lr_clf.best_estimator_
            train_class_preds = optimized_clf.predict(X_train)
            test_class_preds = optimized_clf.predict(X_test)
            train_accuracy_lr = accuracy_score(train_class_preds,Y_train)
            test_accuracy_lr = accuracy_score(test_class_preds,Y_test)

            print("The accuracy on train data is ", train_accuracy_lr)
            print("The accuracy on test data is ", test_accuracy_lr)

            
        except Exception as e:
            raise CustomeException(e,sys)