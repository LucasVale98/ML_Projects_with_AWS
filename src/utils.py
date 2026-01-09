import os
import sys
import pickle
import numpy as np
import pandas as pd
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
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Trains and evaluates multiple models, returning their R2 scores."""
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square
        return model_report
    except Exception as e:
        raise CustomException(e, sys)