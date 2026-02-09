import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import r2_score

# Utility functions for saving objects and evaluating models

def save_object(file_path, obj):
    """Saves a Python object to the specified file path using pickle."""
    try:
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            # Implementando o Grid Search para Hyperparameter Tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Define o modelo com os melhores parâmetros encontrados
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """Loads a Python object from the specified file path using pickle."""
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)   

